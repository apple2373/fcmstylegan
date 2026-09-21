#!/usr/bin/env python3
"""Patch the nested JiT model registry with smaller B-model patch sizes.

This keeps the upstream JiT repository unchanged until this script is run.
The patch adds JiT-B/8 and JiT-B/4 to JiT/model_jit.py.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "JiT" / "model_jit.py"


B8_FUNCTION = """def JiT_B_8(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=8, **kwargs)

"""

B4_FUNCTION = """def JiT_B_4(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=4, **kwargs)

"""


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Nested JiT model file not found: {MODEL_FILE}")

    source = MODEL_FILE.read_text(encoding="utf-8")
    original = source

    if "def JiT_B_8(" not in source:
        marker = "def JiT_B_32(**kwargs):\n"
        if marker not in source:
            raise RuntimeError("Could not find the JiT-B model factory section")
        source = source.replace(marker, B8_FUNCTION + marker, 1)

    if "def JiT_B_4(" not in source:
        marker = "def JiT_B_32(**kwargs):\n"
        if marker not in source:
            raise RuntimeError("Could not find the JiT-B model factory section")
        source = source.replace(marker, B4_FUNCTION + marker, 1)

    if "'JiT-B/8': JiT_B_8," not in source:
        marker = "    'JiT-B/32': JiT_B_32,\n"
        if marker not in source:
            raise RuntimeError("Could not find the JiT model registry")
        source = source.replace(
            marker,
            "    'JiT-B/8': JiT_B_8,\n"
            "    'JiT-B/4': JiT_B_4,\n" + marker,
            1,
        )

    if source == original:
        print(f"JiT-B/8 and JiT-B/4 are already registered in {MODEL_FILE}")
        return

    MODEL_FILE.write_text(source, encoding="utf-8")
    print(f"Patched {MODEL_FILE}")
    print("Added model names: JiT-B/8, JiT-B/4")


if __name__ == "__main__":
    main()
