"""Export the served BERTweet+LoRA model to an optimized ONNX runtime format.

Two model-level optimizations, chained:
1. Format optimization — merge the LoRA adapter into the base weights (no more
   PEFT wrapper / adapter matmuls at inference time), then export the merged
   model to ONNX via `optimum` so it can be run through onnxruntime instead of
   eager PyTorch. (A hand-rolled `torch.onnx.export` was tried first and
   silently produced input-independent logits — transformers' newer
   `masking_utils` builds the attention mask through data-dependent Python
   control flow that the legacy TorchScript tracer freezes as constants from
   the dummy input. `optimum`'s exporter routes through the older, ONNX-safe
   `modeling_attn_mask_utils` path and was verified to reproduce the PyTorch
   model's logits exactly.)
2. Quantization — dynamically quantize the ONNX model's weights from FP32 to
   INT8 (onnxruntime's dynamic quantization; activations stay float and are
   quantized on the fly per batch).

Run from the repo root:
    python milestone_3/export_onnx.py
"""

import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
LORA_PATH = REPO_ROOT / "training" / "bertweet_lora_final_fulldata"
BASE_MODEL_NAME = "vinai/bertweet-base"

MERGED_DIR = Path(__file__).resolve().parent / "merged_model"
OUT_DIR = Path(__file__).resolve().parent / "onnx_model"
FP32_ONNX_PATH = OUT_DIR / "model.onnx"
INT8_ONNX_PATH = OUT_DIR / "model_int8.onnx"

# Tokenizer files that `training/bertweet_lora_final_fulldata` carries in the
# canonical fast-tokenizer format. `save_pretrained` on this tokenizer class
# rewrites vocab.txt/bpe.codes into a format its own loader can't read back
# (see the repo README's "Notes" section), so we copy the source files
# directly instead of round-tripping them through save_pretrained.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "bpe.codes",
    "added_tokens.json",
]


def merge_lora() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(LORA_PATH), normalization=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    peft_model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    merged = peft_model.merge_and_unload()
    merged.eval()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))


def export_onnx() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(str(MERGED_DIR), export=True)
    model.save_pretrained(str(OUT_DIR))

    for name in TOKENIZER_FILES:
        shutil.copy(LORA_PATH / name, OUT_DIR / name)

    print(f"wrote {FP32_ONNX_PATH} ({FP32_ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


def quantize() -> None:
    # per_channel=True matters here: per-tensor dynamic quantization (the
    # default) collapsed this model's predictions to near-constant output
    # regardless of input (verified against the FP32 ONNX and PyTorch
    # models, which agree with each other). Per-channel quantization gives
    # each output channel its own scale/zero-point instead of one shared
    # across the whole weight matrix, which fixed it.
    quantize_dynamic(
        model_input=str(FP32_ONNX_PATH),
        model_output=str(INT8_ONNX_PATH),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"wrote {INT8_ONNX_PATH} ({INT8_ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    merge_lora()
    export_onnx()
    quantize()


if __name__ == "__main__":
    main()
