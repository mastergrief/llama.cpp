#!/usr/bin/env python3
"""Losslessly repack Q2_0 (2-bit-slot ternary) tensors to Q1_75 (base-3, 1.75 bpw).

Q2_0 block:  fp16 d + 32 bytes (128 x 2-bit codes, {0,1,2} used; 3 would be +2)
Q1_75 block: fp16 d + 26 bytes (two 13-byte halves, 64 base-3 digits each,
             5 digits/byte, last digit slot of each half unused)

Same scales, same ternary values -> bit-identical effective weights.
Aborts if any code 3 (+2) is found (model would not be purely ternary).

Usage: python3 repack_q2_0_to_q1_75.py input.gguf output.gguf [--keep-embd]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# use the repo's gguf-py (has Q1_75 registered)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def repack_tensor_data(raw: np.ndarray) -> np.ndarray:
    """Transcode flat uint8 Q2_0 block bytes -> Q1_75 block bytes."""
    blocks = raw.reshape(-1, 34)
    d = blocks[:, :2]
    qs = blocks[:, 2:]  # (nb, 32)

    # unpack 4 x 2-bit codes per byte -> (nb, 128), element j at byte j//4, bits (j%4)*2
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes = (qs[:, :, None] >> shifts[None, None, :]) & 0x3  # (nb, 32, 4)
    codes = codes.reshape(-1, 128).astype(np.uint16)

    n3 = int((codes == 3).sum())
    if n3 > 0:
        raise SystemExit(f"ABORT: found {n3} code-3 (+2) values -- source is not purely ternary")

    # per 64-digit half: pad to 65 digits (13 bytes x 5), dot with powers of 3
    halves = codes.reshape(-1, 2, 64)
    padded = np.zeros((halves.shape[0], 2, 65), dtype=np.uint16)
    padded[:, :, :64] = halves
    packed = (padded.reshape(-1, 2, 13, 5) * POW3[None, None, None, :]).sum(axis=-1)
    assert packed.max() <= 242
    packed = packed.astype(np.uint8).reshape(-1, 26)  # (nb, 26)

    out = np.concatenate([d, packed], axis=1)  # (nb, 28)

    # self-check: decode both representations for a sample of blocks, must match exactly
    sample = np.random.default_rng(17).integers(0, len(blocks), size=min(4096, len(blocks)))
    dec_src = codes[sample].astype(np.int8) - 1
    b = packed[sample].reshape(-1, 2, 13)
    digits = (b[:, :, :, None] // POW3[None, None, None, :].astype(np.uint8)) % 3
    dec_dst = digits.reshape(len(sample), 2, 65)[:, :, :64].reshape(len(sample), 128).astype(np.int8) - 1
    assert np.array_equal(dec_src, dec_dst), "round-trip mismatch"

    return out.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output')
    args = ap.parse_args()

    reader = GGUFReader(args.input)
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = GGUFWriter(args.output, arch)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    plan = []
    saved = 0
    for tensor in reader.tensors:
        if tensor.tensor_type == GGMLQuantizationType.Q2_0 and 'token_embd' not in tensor.name:
            plan.append((tensor, True))
            saved += tensor.n_bytes - (tensor.n_bytes // 34) * 28
        else:
            plan.append((tensor, False))
    repacked = {}

    print(f"repacking {sum(1 for _, r in plan if r)} tensors, passing through "
          f"{sum(1 for _, r in plan if not r)}; projected saving {saved/1e9:.2f} GB")

    for tensor, repack in plan:
        if repack:
            data = repack_tensor_data(np.ascontiguousarray(tensor.data).reshape(-1))
            byte_shape = tensor.data.shape
            if len(byte_shape) >= 2:
                new_shape = byte_shape[:-1] + ((byte_shape[-1] // 34) * 28,)
            else:
                new_shape = ((byte_shape[0] // 34) * 28,)
            writer.add_tensor_info(tensor.name, new_shape, np.uint8, len(data),
                                   raw_dtype=GGMLQuantizationType.Q1_75)
            repacked[tensor.name] = data
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                   tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor, repack in plan:
        buf = repacked[tensor.name] if repack else tensor.data
        writer.write_tensor_data(buf, tensor_endianess=reader.endianess)
        print(f"  wrote {tensor.name} ({'Q1_75' if repack else tensor.tensor_type.name})", flush=True)

    writer.close()
    print("done")


if __name__ == '__main__':
    main()
