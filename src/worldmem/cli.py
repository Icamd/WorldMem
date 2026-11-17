"""Commands for building, querying, and evaluating WorldMem memories."""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .api import PROFILES, WorldMem
from .clients import EmbeddingClient, LLMClient
from .config import MemoryConfig
from .runner import build_dataset, retrieve_dataset
from .storage import write_json


def parser():
    root = argparse.ArgumentParser(prog="worldmem", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ["build", "retrieve", "ask"]:
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", required=True, help="Experiment preset JSON")
        cmd.add_argument("--memory-dir", help="Override the memory storage directory")
        cmd.add_argument("--backend", choices=["hf", "vllm", "openai_api"])
        cmd.add_argument("--llm-model", help="HF path/model ID, or API served model name")
        cmd.add_argument("--base-url", help="OpenAI-compatible endpoint, including /v1")
        cmd.add_argument("--embedding-backend", choices=["local", "api", "hash"])
        cmd.add_argument("--embedding-model")
        cmd.add_argument("--embedding-device", default=None)
        cmd.add_argument("--device", help="HF device_map, for example cpu or auto")
        cmd.add_argument("--verbose", action="store_true")
        if name in {"build", "retrieve"}:
            cmd.add_argument(
                "--data",
                required=True,
                help="Dataset JSON; PerLTQA uses distinct memory and QA files",
            )
            cmd.add_argument(
                "--limit", type=int, default=0, help="Limit conversations/characters; 0 means all"
            )
            cmd.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
        if name == "build":
            cmd.add_argument(
                "--session-limit", type=int, default=None, help="LoCoMo sessions per conversation"
            )
            cmd.add_argument(
                "--overwrite",
                action="store_true",
                help="Replace selected owners after a successful staged rebuild",
            )
        elif name == "retrieve":
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--qa-limit", type=int, default=0)
            cmd.add_argument("--workers", type=int, default=1)
            cmd.add_argument(
                "--categories",
                help="Comma-separated LoCoMo category numbers or PerLTQA category names",
            )
            cmd.add_argument(
                "--experiment", choices=["full", "no_mental", "no_situation"], default="full"
            )
            cmd.add_argument(
                "--full-results",
                action="store_true",
                help="Include source evidence and stage traces",
            )
        else:
            cmd.add_argument("--owner", required=True)
            cmd.add_argument("--query", required=True)
            cmd.add_argument("--output", help="Optional JSON output including evidence and traces")
    demo = commands.add_parser(
        "demo", help="CPU-only synthetic integration demo; no model downloads or API calls"
    )
    demo.add_argument("--output-dir", default="outputs/demo")
    score = commands.add_parser("score", help="Offline rule-based metrics")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    return root


def load_preset(args):
    preset = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if preset.get("profile") not in PROFILES:
        raise ValueError(f"Invalid profile: {preset.get('profile')}")
    if preset.get("dataset") != preset["profile"].split("_")[0]:
        raise ValueError("Dataset and retrieval profile do not match")
    preset.setdefault("memory", {})
    preset["memory"].setdefault("storage_path", f"outputs/{preset['profile']}/memories")
    if args.memory_dir:
        preset["memory"]["storage_path"] = args.memory_dir
    llm = preset["llm"]
    backend = (
        args.backend or llm["build_backend" if args.command == "build" else "retrieval_backend"]
    )
    model = args.llm_model or (llm["model"] if backend == "hf" else llm["served_model"])
    llm["resolved_backend"], llm["resolved_model"] = (backend, model)
    if args.base_url:
        llm["base_url"] = args.base_url
    if backend == "openai_api":
        llm["base_url"] = args.base_url or os.environ.get("OPENAI_BASE_URL") or llm.get("base_url")
    for key, arg in [
        ("backend", args.embedding_backend),
        ("model", args.embedding_model),
        ("device", args.embedding_device),
    ]:
        if arg is not None:
            preset["embedding"][key] = arg
    preset["memory"]["llm_model"] = model
    if args.device:
        llm["device"] = args.device
    return preset


def main(argv=None):
    args = parser().parse_args(argv)
    load_dotenv(Path.cwd() / ".env")
    cache_root = Path.cwd() / ".cache"
    for key, location in {
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root,
    }.items():
        os.environ.setdefault(key, str(location))
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for key in ["limit", "qa_limit", "workers", "session_limit"]:
        value = getattr(args, key, None)
        if value is not None and (
            value < 0 or (key in {"workers", "session_limit"} and value == 0)
        ):
            raise ValueError(f"{key} has an invalid value: {value}")
    if args.command == "demo":
        from .demo import run_demo

        print(json.dumps(run_demo(Path(args.output_dir)), ensure_ascii=False, indent=2))
        return
    if args.command == "score":
        from .evaluation.metrics import evaluate_results

        report = evaluate_results(json.loads(Path(args.input).read_text(encoding="utf-8")))
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    preset = load_preset(args)
    llm_options, emb_options = (preset["llm"], preset["embedding"])
    llm = LLMClient(
        backend=llm_options["resolved_backend"],
        model=llm_options["resolved_model"],
        base_url=llm_options.get("base_url"),
        device=llm_options.get("device", "auto"),
        reasoning_effort=llm_options.get("reasoning_effort", "minimal"),
        min_completion_tokens=llm_options.get("min_completion_tokens", 4096),
    )
    embeddings = None
    try:
        embeddings = EmbeddingClient(
            model=emb_options["model"],
            backend=emb_options.get("backend", "local"),
            local_device=emb_options.get("device", "cpu"),
            base_url=emb_options.get("base_url") or os.environ.get("EMBEDDING_BASE_URL") or None,
        )
        if args.command == "ask":
            memory = WorldMem(
                MemoryConfig.from_dict(preset["memory"]),
                llm_client=llm,
                embedding_client=embeddings,
                profile=preset["profile"],
                retrieval_options=preset.get("retrieval", {}),
            )
            result = memory.ask(args.owner, args.query)
            if args.output:
                write_json(args.output, result)
            print(result["response"])
        else:
            dataset = json.loads(Path(args.data).read_text(encoding="utf-8"))
            if args.command == "build":
                result = build_dataset(
                    dataset,
                    preset,
                    llm,
                    embeddings,
                    limit=args.limit,
                    session_limit=args.session_limit,
                    resume=args.resume,
                    overwrite=args.overwrite,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                result = retrieve_dataset(
                    dataset,
                    preset,
                    llm,
                    embeddings,
                    output=args.output,
                    experiment=args.experiment,
                    limit=args.limit,
                    qa_limit=args.qa_limit,
                    categories=[x.strip() for x in args.categories.split(",")]
                    if args.categories
                    else None,
                    resume=args.resume,
                    workers=args.workers,
                    full_results=args.full_results,
                )
                print(
                    f"Saved {sum((len(rows) for rows in result.values()))} answers to {args.output}"
                )
    finally:
        llm.close()
        if embeddings is not None:
            embeddings.close()
