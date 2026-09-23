# Milestone 3: Model-Level Optimization (Week 5)

> System/infra-level optimizations (dynamic batching, Redis caching) that serve this
> optimized model are in `backend/` — see the top-level README's "Milestone 3: Model &
> Infrastructure Optimization" section for the full picture, architecture diagram, and
> how the two weeks combine into the deployed `docker compose` stack.

Two chained model-level optimizations applied to the served model
(`training/bertweet_lora_final_fulldata`, a LoRA fine-tune of `vinai/bertweet-base`):

1. **Model format optimization** — merge the LoRA adapter into the base weights
   (`peft`'s `merge_and_unload`, eliminating the adapter matmuls at inference
   time) and export the merged model to **ONNX** via `optimum`, so inference
   runs through `onnxruntime` instead of eager PyTorch.
2. **Quantization** — dynamically quantize the ONNX model's weights from
   FP32 to **INT8** (`onnxruntime.quantization.quantize_dynamic`, per-channel).

## Files

- `export_onnx.py` — does both steps above; run once to produce
  `merged_model/` (intermediate, gitignored) and `onnx_model/model.onnx` +
  `onnx_model/model_int8.onnx` (gitignored — regenerate locally, they're
  520 MB / 136 MB and don't belong in git).
- `optimized_model.py` — `load_model()` / `predict_one()` / `predict_batch()`
  for the optimized model, mirroring `milestone_2/backend/app/model.py`'s
  interface for the naive one.
- `_bench_worker.py` — loads one model variant and benchmarks it; run as a
  subprocess so each variant's memory measurement isn't contaminated by the
  other variant's weights also sitting in the process.
- `benchmark.py` — the required benchmarking script. Spawns both workers,
  sends the same batch of tweets (sampled from `train.csv`) to each one
  request at a time, and reports latency, throughput, memory, and accuracy.

## Running it

```
pip install -r milestone_3/requirements-bench.txt
python milestone_3/export_onnx.py       # ~5 min on CPU, needs the LoRA checkpoint
python milestone_3/benchmark.py --num-samples 200
```

(`requirements.txt` alone — no `psutil` — is what the Docker build in `milestone_3/backend/Dockerfile`
installs to run just `export_onnx.py`; `requirements-bench.txt` adds `psutil` for `benchmark.py`,
which isn't needed outside this local benchmark.)

## Results (200 requests, single-request latency, CPU-only, per-channel INT8)

| metric | naive (FP32 PyTorch+PEFT) | optimized (ONNX INT8) | ratio |
|---|---:|---:|---:|
| load time (s) | 1.4 – 9.3* | 0.5 – 0.9* | faster, varies with disk cache |
| model footprint (RSS delta, MB) | ~898 | ~501 | 1.79x smaller |
| peak RSS (MB) | ~998 | ~595 | 1.68x smaller |
| mean latency (ms) | 82.8 – 128.7 | 12.9 – 16.6 | 6.4 – 7.8x faster |
| p50 latency (ms) | 83.3 – 113.9 | 12.4 – 16.2 | 6.7 – 7.0x faster |
| p95 latency (ms) | 123.5 – 263.6 | 18.6 – 24.2 | 6.7 – 10.9x faster |
| throughput (req/s) | 7.8 – 12.1 | 60.3 – 77.6 | 6.4 – 7.8x |
| accuracy vs. label | 88.5% | 85.0% | -3.5 pts |
| served-artifact disk size | 3.6 MB adapter + ~540 MB shared base checkpoint | 136 MB, self-contained | ~4x smaller |

*Latency/throughput/load-time ranges are across two runs (`--num-samples 200`,
same seed) on the same otherwise-idle machine — memory footprint, agreement,
and accuracy were identical across runs since predictions are deterministic.

Prediction agreement between the two models on the same 200 requests: **91.5%**.

Full raw output: `results/benchmark_summary.json` (regenerated on each run).

### Notes on the accuracy numbers

`train.csv` was part of what `bertweet_lora_final_fulldata` was fine-tuned
on, so "accuracy vs. label" here is a memorization check, not a held-out
generalization estimate — treat the **91.5% agreement** between the two
models as the more meaningful signal for "did quantization change behavior."
A 3.5-point accuracy gap and 8.5% disagreement rate against a 7.8x latency
win and ~4x smaller self-contained artifact is a reasonable trade for this
use case; a held-out validation split would be needed before calling this
conclusively "acceptable."

### A quantization pitfall worth documenting

The first working ONNX export, once dynamically quantized with
onnxruntime's default (per-tensor) settings, silently collapsed to
near-input-independent predictions — confidences clustered around 0.87-0.94
regardless of the tweet, with strong disaster examples flipping to "not
disaster." The FP32 ONNX model (verified to exactly reproduce the PyTorch
model's logits) was fine; only the INT8 step broke it. Restricting
quantization to `MatMul` ops didn't fix it. Switching to **per-channel**
quantization (`per_channel=True`, giving each output channel its own
scale/zero-point instead of one shared across the whole weight matrix) fixed
it completely — see the comment in `export_onnx.py`'s `quantize()`.

A second, earlier pitfall: a hand-rolled `torch.onnx.export` (before
switching to `optimum`) also produced input-independent logits — newer
`transformers` versions build the attention mask through data-dependent
Python control flow (`masking_utils`) that the legacy TorchScript tracer
freezes as constants from whatever the dummy export input looked like.
`optimum`'s exporter routes through the older, ONNX-export-safe
`modeling_attn_mask_utils` code path and was verified correct against the
PyTorch model before quantizing.
