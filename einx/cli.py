# -*- coding: utf-8 -*-
"""EINX command-line interface.

Subcommands (installed as console scripts by pyproject.toml):
    einx-tokenizer   — train / inspect a tokenizer
    einx-train       — train a model
    einx-generate    — generate text from a checkpoint
    einx-evaluate    — evaluate a checkpoint
    einx-serve       — start the HTTP API server

Each is also accessible via ``python -m einx.cli <subcommand>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# einx-tokenizer
# ---------------------------------------------------------------------------


def tokenizer_main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="einx-tokenizer", description="EINX tokenizer tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a BPE tokenizer on a corpus")
    p_train.add_argument("--corpus", required=True, help="Path to JSONL corpus (with 'text' field)")
    p_train.add_argument("--output", required=True, help="Output tokenizer JSON path")
    p_train.add_argument("--vocab-size", type=int, default=4096)
    p_train.add_argument("--quiet", action="store_true")

    p_info = sub.add_parser("info", help="Show tokenizer info")
    p_info.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON")

    p_encode = sub.add_parser("encode", help="Encode text -> token IDs")
    p_encode.add_argument("--tokenizer", required=True)
    p_encode.add_argument("--text", required=True)

    p_decode = sub.add_parser("decode", help="Decode token IDs -> text")
    p_decode.add_argument("--tokenizer", required=True)
    p_decode.add_argument("--ids", required=True, help="Comma-separated token IDs")

    args = parser.parse_args(argv)
    _setup_logging("WARNING" if args.quiet else "INFO")

    from einx.tokenizer.bpe import BPETokenizer

    if args.command == "train":
        from einx.data.dataset import load_jsonl
        records = load_jsonl(args.corpus)
        texts = [r["text"] for r in records]
        tok = BPETokenizer()
        meta = tok.train(texts, vocab_size=args.vocab_size, verbose=not args.quiet)
        tok.save(args.output)
        print(json.dumps(meta, indent=2))
        return 0

    if args.command == "info":
        tok = BPETokenizer.load(args.tokenizer)
        print(f"version: {tok.VERSION}")
        print(f"vocab_size: {tok.vocab_size()}")
        print(f"merges: {len(tok.merges)}")
        print(f"special tokens: {tok.special.all}")
        print(f"first 10 tokens: {tok.vocab[:10]}")
        return 0

    if args.command == "encode":
        tok = BPETokenizer.load(args.tokenizer)
        ids = tok.encode(args.text)
        print(json.dumps(ids))
        return 0

    if args.command == "decode":
        tok = BPETokenizer.load(args.tokenizer)
        ids = [int(i.strip()) for i in args.ids.split(",") if i.strip()]
        print(tok.decode(ids))
        return 0

    return 1


# ---------------------------------------------------------------------------
# einx-train
# ---------------------------------------------------------------------------


def train_main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="einx-train", description="Train an EINX model")
    parser.add_argument("--model-config", default="", help="Path to model YAML (default: einx-experimental)")
    parser.add_argument("--model-name", default="einx-experimental", help="Built-in model name")
    parser.add_argument("--training-config", default="", help="Path to training YAML")
    parser.add_argument("--training-profile", default="experimental", help="Built-in profile name")
    parser.add_argument("--tokenizer", default="data/tokenized/einx-bpe.json")
    parser.add_argument("--dataset", default="data/processed/train.jsonl")
    parser.add_argument("--val-dataset", default="data/processed/val.jsonl")
    parser.add_argument("--resume", default="", help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=0, help="Override max_steps (0 = use config)")
    parser.add_argument("--prepare-synthetic", type=int, default=0,
                        help="If >0, write N synthetic records to data/processed/ and use them")
    # ----- Build 2.1: hardware-adaptive flags (spec §3, §4, §22)
    parser.add_argument("--device", default="auto",
                        help="Compute device: auto | cpu | cuda | cuda:0 | mps (default: auto)")
    parser.add_argument("--precision", default="auto",
                        help="Precision: auto | fp32 | fp16 | bf16 (default: auto)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (default: off — adds warmup cost)")
    args = parser.parse_args(argv)
    _setup_logging()

    from einx.config import get_model_config, get_training_config, EINXModelConfig, TrainingConfig, RuntimeConfig
    from einx.data.dataset import load_jsonl, TokenisedDataset
    from einx.data.synthetic import write_synthetic_corpus
    from einx.tokenizer.bpe import BPETokenizer
    from einx.model.transformer import EINXTransformer
    from einx.training.trainer import EINXTrainer

    # Optional: generate synthetic corpus for first-run
    if args.prepare_synthetic > 0:
        from einx.data.dataset import write_jsonl, train_val_test_split
        n = args.prepare_synthetic
        from einx.data.synthetic import generate_synthetic_corpus
        records = [{"text": t} for t in generate_synthetic_corpus(n)]
        train, val, _ = train_val_test_split(records, val_ratio=0.1, test_ratio=0.05)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        write_jsonl(train, args.dataset)
        write_jsonl(val, args.val_dataset)
        write_jsonl(_, "data/processed/test.jsonl")
        # Train tokenizer too if missing
        if not Path(args.tokenizer).exists():
            tok = BPETokenizer()
            tok.train([r["text"] for r in train], vocab_size=4096)
            tok.save(args.tokenizer)

    # Configs
    if args.model_config:
        model_cfg = EINXModelConfig.from_yaml(args.model_config)
    else:
        model_cfg = get_model_config(args.model_name)

    if args.training_config:
        train_cfg = TrainingConfig.from_yaml(args.training_config)
    else:
        train_cfg = get_training_config(args.training_profile)
    if args.steps > 0:
        train_cfg.max_steps = args.steps
    if args.resume:
        train_cfg.resume_from = args.resume
    train_cfg.seed = args.seed
    train_cfg.tokenizer_path = args.tokenizer
    train_cfg.dataset_path = args.dataset
    train_cfg.val_dataset_path = args.val_dataset

    # Build the runtime config (spec §16) — carries device, precision,
    # distributed, backend, compile settings.  Default is "auto" for
    # everything, meaning EINX decides based on actual hardware.
    runtime_cfg = RuntimeConfig(
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    # Load tokenizer + datasets
    tokenizer = BPETokenizer.load(args.tokenizer)
    train_records = load_jsonl(args.dataset)
    train_texts = [r["text"] for r in train_records]
    train_ds = TokenisedDataset(train_texts, tokenizer, context_length=model_cfg.max_context_length)

    val_ds = None
    if Path(args.val_dataset).exists():
        val_records = load_jsonl(args.val_dataset)
        val_texts = [r["text"] for r in val_records]
        val_ds = TokenisedDataset(val_texts, tokenizer, context_length=model_cfg.max_context_length)

    # Build model
    model = EINXTransformer(model_cfg)
    print(f"Model: {model}")
    print(f"  params: {model.n_params:,}")

    # Train — pass the runtime config so the trainer can resolve "auto"
    # against actual hardware, print the hardware banner, and use the
    # right device + precision.
    trainer = EINXTrainer(model, train_cfg, train_ds, val_ds,
                         tokenizer=tokenizer, runtime_config=runtime_cfg)
    print(f"Trainer: {trainer}")
    result = trainer.train()
    print(f"\nTraining complete. Final step: {result['final_step']}, best val loss: {result['best_val_loss']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# einx-generate
# ---------------------------------------------------------------------------


def generate_main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="einx-generate", description="Generate text from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream", action="store_true", help="Stream tokens")
    args = parser.parse_args(argv)
    _setup_logging("WARNING")

    from einx.inference.generator import EINXGenerator, GenerationConfig

    gen = EINXGenerator.from_checkpoint(args.checkpoint, args.tokenizer)
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed or None,
    )
    if args.stream:
        print(args.prompt, end="", flush=True)
        for chunk in gen.stream(args.prompt, cfg):
            print(chunk, end="", flush=True)
        print()
    else:
        result = gen.generate(args.prompt, cfg)
        print(result.text)
        print(f"\n[{result.n_output_tokens} tokens, {result.elapsed_seconds:.2f}s, {result.tokens_per_second:.1f} tok/s]",
              file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# einx-evaluate
# ---------------------------------------------------------------------------


def evaluate_main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="einx-evaluate", description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="experiments/eval_results.json")
    args = parser.parse_args(argv)
    _setup_logging()

    from einx.evaluation.evaluator import run_evaluation
    result = run_evaluation(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        eval_dataset_path=args.dataset,
        output_path=args.output,
    )
    print(json.dumps(result.metrics, indent=2))
    return 0


# ---------------------------------------------------------------------------
# einx-serve
# ---------------------------------------------------------------------------


def serve_main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="einx-serve", description="Start the EINX HTTP API")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path (optional — runs without model if omitted)")
    parser.add_argument("--tokenizer", default="", help="Tokenizer path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    _setup_logging()

    import uvicorn
    from einx.api.server import create_app
    app = create_app(args.checkpoint or None, args.tokenizer or None)
    print(f"EINX API starting on http://{args.host}:{args.port}")
    print("  GET  /health     — service health")
    print("  GET  /model      — model metadata")
    print("  POST /generate   — text generation")
    print("  POST /chat       — chat completion")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# ---------------------------------------------------------------------------
# einx-hardware (Build 2.1 — spec §2)
# ---------------------------------------------------------------------------


def hardware_main(argv=None) -> int:
    """Print the EINX hardware report (spec §2).

    Detects CPU, CUDA, MPS, GPU count, GPU names + memory, recommended
    precision, and distributed-training capability.  Never claims CUDA
    exists when it doesn't.
    """
    parser = argparse.ArgumentParser(
        prog="einx-hardware",
        description="Print the EINX hardware report (device, precision, distributed capability)",
    )
    parser.add_argument("--device", default="auto",
                        help="Preference: auto | cpu | cuda | mps (default: auto)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    from einx.utils.hardware import get_hardware_report, format_hardware_report
    report = get_hardware_report(args.device)
    if args.json:
        import json
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_hardware_report(report))
    return 0


# ---------------------------------------------------------------------------
# einx-distributed-train (Build 2.1 — spec §8)
# ---------------------------------------------------------------------------


def distributed_train_main(argv=None) -> int:
    """Launch distributed training via torchrun (spec §8).

    Wraps ``torchrun --nproc_per_node=N einx-train ...`` so users don't
    need to remember the torchrun syntax.  Auto-detects GPU count when
    ``--nproc`` is not specified.
    """
    parser = argparse.ArgumentParser(
        prog="einx-distributed-train",
        description="Launch distributed training (wraps torchrun)",
    )
    parser.add_argument("--nproc", type=int, default=0,
                        help="Number of processes (default: auto-detect from GPU count)")
    parser.add_argument("--strategy", default="ddp",
                        choices=["ddp", "fsdp"],
                        help="Distributed strategy (default: ddp)")
    parser.add_argument("--port", type=int, default=29500,
                        help="Master port for torchrun (default: 29500)")
    parser.add_argument("--training-args", default="",
                        help="Args to pass through to einx-train (quote them)")
    args, train_args = parser.parse_known_args(argv)

    from einx.utils.hardware import is_cuda_available, _cuda_device_count
    # Determine nproc
    nproc = args.nproc
    if nproc <= 0:
        if is_cuda_available():
            nproc = _cuda_device_count()
            if nproc < 2:
                print(f"ERROR: distributed training requires >=2 CUDA GPUs. "
                      f"Detected: {nproc}.", file=sys.stderr)
                return 1
        else:
            print("ERROR: distributed training requires CUDA GPUs. "
                  "Detected: 0. For CPU distributed infrastructure tests, "
                  "run the test suite instead.", file=sys.stderr)
            return 1

    print(f"Launching distributed training:")
    print(f"  Strategy:  {args.strategy}")
    print(f"  Processes: {nproc}")
    print(f"  Port:       {args.port}")
    print(f"  Train args: {train_args}")
    print()

    # Build the torchrun command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={args.port}",
        "-m", "einx.cli", "train",
    ] + train_args
    # Pass the strategy via env var so the trainer picks it up
    import os
    os.environ["EINX_DIST_STRATEGY"] = args.strategy

    import subprocess
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# python -m einx.cli <subcommand>
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    if len(sys.argv) < 2 if argv is None else len(argv) < 1:
        print("usage: python -m einx.cli {tokenizer|train|generate|evaluate|serve|hardware|distributed-train} ...",
              file=sys.stderr)
        return 2
    cmd = (argv or sys.argv[1:])[0]
    rest = (argv or sys.argv[1:])[1:]
    if cmd == "tokenizer":
        return tokenizer_main(rest)
    if cmd == "train":
        return train_main(rest)
    if cmd == "generate":
        return generate_main(rest)
    if cmd == "evaluate":
        return evaluate_main(rest)
    if cmd == "serve":
        return serve_main(rest)
    if cmd == "hardware":
        return hardware_main(rest)
    if cmd == "distributed-train":
        return distributed_train_main(rest)
    if cmd in ("-h", "--help", "help"):
        print("EINX CLI — available subcommands:")
        print("  tokenizer          Train / inspect / encode / decode a BPE tokenizer")
        print("  train              Train an EINX model (--device auto|cpu|cuda)")
        print("  distributed-train  Launch DDP/FSDP training via torchrun")
        print("  generate           Generate text from a checkpoint")
        print("  evaluate           Evaluate a checkpoint (loss, perplexity, latency)")
        print("  serve              Start the HTTP API server")
        print("  hardware           Print the hardware report (device, precision, GPUs)")
        return 0
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
