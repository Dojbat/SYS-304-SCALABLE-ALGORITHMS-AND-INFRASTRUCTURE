"""Merge a LoRA adapter into BERTweet, export to ONNX, quantize to INT8.

The same three steps as milestone_3/export_onnx.py (see that file for why
`optimum` is used instead of torch.onnx.export, and why quantization must be
per-channel), but for any adapter directory instead of the one hard-coded
checkpoint.

Runs inside the `exporter` container (Dockerfile.export), which pins the
exact torch/transformers/optimum versions that Milestone 3 verified produce
a correct export:

    docker compose --profile tools run --rm exporter \
        --adapter /models/_staging/v2/adapter --out /models/_staging/v2
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

BASE_MODEL_NAME = "vinai/bertweet-base"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "bpe.codes",
    "added_tokens.json",
]


def export(adapter_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        merged_dir = Path(tmp) / "merged"
        fp32_dir = Path(tmp) / "fp32"

        base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.eval()
        merged.save_pretrained(str(merged_dir))

        ORTModelForSequenceClassification.from_pretrained(
            str(merged_dir), export=True
        ).save_pretrained(str(fp32_dir))

        int8_path = out_dir / "model_int8.onnx"
        quantize_dynamic(
            model_input=str(fp32_dir / "model.onnx"),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        shutil.copy(fp32_dir / "config.json", out_dir / "config.json")

    for name in TOKENIZER_FILES:
        shutil.copy(adapter_dir / name, out_dir / name)
    print(f"wrote {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
    return int8_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.adapter, args.out)


if __name__ == "__main__":
    main()
