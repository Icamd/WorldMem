"""Dataset orchestration, transactional builds, and resumable benchmark answers."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

from .api import WorldMem
from .config import MemoryConfig
from .construction.builder import MemoryBuilder
from .construction.perltqa import PerLTQABuilder
from .datasets.locomo import build_messages
from .datasets.perltqa import build_perltqa_character_memory
from .datasets.perltqa_input import get_character_name, iter_perltqa_characters, make_owner_id
from .storage import LAYERS, layer_path, read_jsonl, validate_owner, write_json


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def construction_items(dataset, kind, *, limit=0, session_limit=None, owner_id_format="name_key"):
    if kind == "locomo":
        if not isinstance(dataset, list):
            raise ValueError("LoCoMo data must be a JSON list of conversations")
        for idx, item in enumerate(dataset[:limit] if limit else dataset):
            conv = item.get("conversation", {})
            owner = validate_owner(f"{conv.get('speaker_a', 'speaker_a')}_{idx}")
            messages = build_messages(conv, session_limit=session_limit)
            if not messages:
                raise ValueError(f"No messages in conversation {idx}")
            yield (
                owner,
                messages,
                dict(character_key=str(idx), character_name=conv.get("speaker_a", "speaker_a")),
            )
    elif kind == "perltqa":
        items = iter_perltqa_characters(dataset)
        for idx, (key, character) in enumerate(items[:limit] if limit else items):
            name = get_character_name(character, fallback=key)
            owner = make_owner_id(key, idx, name, owner_id_format)
            yield (validate_owner(owner), character, dict(character_key=key, character_name=name))
    else:
        raise ValueError(f"Unknown dataset: {kind}")


def build_dataset(
    dataset, preset, llm, embeddings, *, limit=0, session_limit=None, resume=True, overwrite=False
):
    config = MemoryConfig.from_dict(preset["memory"])
    root = Path(config.storage_path)
    staging = root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "owners.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary = []
    options = preset.get("construction", {})
    owners = set()
    for owner, payload, identity in construction_items(
        dataset,
        preset["dataset"],
        limit=limit,
        session_limit=session_limit,
        owner_id_format=preset.get("owner_id_format", "name_key"),
    ):
        if owner in owners:
            raise ValueError(f"Multiple dataset records resolve to owner {owner}")
        owners.add(owner)
        signature = fingerprint(
            dict(
                payload=payload,
                profile=preset["profile"],
                memory=preset["memory"],
                llm=preset["llm"],
                embedding=preset["embedding"],
                construction=options,
            )
        )
        marker = root / "checkpoints" / f"{owner}.json"
        checkpoint = json.loads(marker.read_text()) if marker.exists() else {}
        paths = {layer: layer_path(root, owner, layer) for layer in LAYERS}
        complete = checkpoint.get("fingerprint") == signature and all(
            (
                path.exists() and checkpoint.get("hashes", {}).get(layer) == file_hash(path)
                for layer, path in paths.items()
            )
        )
        if resume and complete and (not overwrite):
            manifest[owner] = dict(identity, owner_id=owner)
            write_json(manifest_path, manifest)
            summary.append(dict(owner=owner, status="skipped"))
            continue
        if checkpoint and checkpoint.get("fingerprint") != signature and (not overwrite):
            raise ValueError(
                f"Input/model/config changed for {owner}; use --overwrite to replace this owner's memory"
            )
        if any((path.exists() for path in paths.values())) and (not checkpoint) and (not overwrite):
            raise FileExistsError(
                f"Existing memory for {owner} has no matching checkpoint; use a new memory directory or --overwrite"
            )
        with TemporaryDirectory(prefix=f"{owner}-", dir=staging) as directory:
            stage_config = replace(config, storage_path=directory)
            if preset["dataset"] == "locomo":
                builder = MemoryBuilder(stage_config, llm, embeddings)
                result = builder.build_world_memory_from_messages(
                    owner, payload, batch_size=options.get("batch_size", 40)
                )
            else:
                builder = PerLTQABuilder(stage_config, llm, embeddings)
                result = build_perltqa_character_memory(
                    builder,
                    owner_id=owner,
                    character_data=payload,
                    character_id=identity["character_key"],
                    character_name=identity["character_name"],
                    enable_dialogue_atomic_backfill=options.get("atomic_backfill", True),
                    annotate_agent_labels=options.get("annotate_agent_labels", True),
                )
            stage_paths = {layer: layer_path(directory, owner, layer) for layer in LAYERS}
            if not all((path.exists() and read_jsonl(path) for path in stage_paths.values())):
                raise RuntimeError(
                    f"Build for {owner} did not produce nonempty, valid memory in every layer"
                )
            hashes = {layer: file_hash(path) for layer, path in stage_paths.items()}
            for layer, path in stage_paths.items():
                paths[layer].parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, paths[layer])
            write_json(marker, dict(fingerprint=signature, hashes=hashes, counts=result))
        manifest[owner] = dict(identity, owner_id=owner)
        write_json(manifest_path, manifest)
        summary.append(dict(owner=owner, status="built", counts=result))
    if not summary:
        raise ValueError("Dataset selection is empty")
    return summary


def qa_items(dataset, preset, *, limit=0, qa_limit=0, categories=None):
    if preset["dataset"] == "locomo":
        for idx, item in enumerate(dataset[:limit] if limit else dataset):
            owner = f"{item.get('conversation', {}).get('speaker_a', 'speaker')}_{idx}"
            rows = item.get("qa", [])
            for qa in rows[:qa_limit] if qa_limit else rows:
                if categories and str(qa.get("category")) not in categories:
                    continue
                yield (str(idx), owner, qa)
    else:
        module = import_module(f"worldmem.retrieval.{preset['profile']}")
        root = Path(preset["memory"]["storage_path"])
        manifest = (
            json.loads((root / "owners.json").read_text())
            if (root / "owners.json").exists()
            else {}
        )
        aliases = {}
        for owner, identity in manifest.items():
            for key in {identity["character_key"], identity["character_name"], owner}:
                aliases.setdefault(key, set()).add(owner)
        flattened = module.flatten_perltqa_qa_dataset(
            dataset,
            owner_id_format=preset.get("owner_id_format", "name_key"),
            character_limit=limit,
            limit_qa_per_character=qa_limit,
        )
        for item in flattened:
            candidates = aliases.get(item["character_key"], set())
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous memory owner for {item['character_key']}")
            owner = next(iter(candidates)) if candidates else item["owner_id"]
            for qa in item["qa"]:
                if categories and qa.get("category") not in categories:
                    continue
                yield (owner, owner, qa)


def retrieve_dataset(
    dataset,
    preset,
    llm,
    embeddings,
    *,
    output,
    experiment="full",
    limit=0,
    qa_limit=0,
    categories=None,
    resume=True,
    workers=1,
    full_results=False,
):
    categories = categories if categories is not None else preset.get("categories")
    config = MemoryConfig.from_dict(preset["memory"])
    output = Path(output)
    selected = list(
        qa_items(dataset, preset, limit=limit, qa_limit=qa_limit, categories=categories)
    )
    memory_hashes = {}
    for owner in sorted({owner for _, owner, _ in selected}):
        memory_hashes[owner] = {}
        for layer in LAYERS:
            path = layer_path(config.storage_path, owner, layer)
            memory_hashes[owner][layer] = file_hash(path) if path.exists() else None
    signature = fingerprint(
        dict(
            dataset=dataset,
            preset=preset,
            experiment=experiment,
            limit=limit,
            qa_limit=qa_limit,
            categories=categories,
            full_results=full_results,
            memory_hashes=memory_hashes,
        )
    )
    metadata_path = output.with_suffix(".run.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() and resume else {}
    if resume and output.exists() and (metadata.get("fingerprint") != signature):
        raise ValueError(
            "Results have a different input/config/memory; select a new --output or use --no-resume"
        )
    results = json.loads(output.read_text()) if resume and output.exists() else {}
    # The answer file is canonical. A crash between the two atomic writes must
    # not skip an answer that never reached disk or duplicate one already saved.
    done = {
        row["_worldmem_id"] for rows in results.values() for row in rows if "_worldmem_id" in row
    }
    jobs = []
    queued = set()
    for group, owner, qa in selected:
        question = qa.get("question", "").strip()
        key = fingerprint(
            dict(
                group=group,
                owner=owner,
                question=question,
                category=qa.get("category"),
                source=qa.get("qa_source_id"),
            )
        )
        if question and key not in done and key not in queued:
            jobs.append((key, group, owner, qa))
            queued.add(key)
    if not jobs and (not done):
        raise ValueError("No QA records selected; check the dataset layout and category filters")
    memory = WorldMem(
        config,
        llm_client=llm,
        embedding_client=embeddings,
        profile=preset["profile"],
        retrieval_options=preset.get("retrieval", {}),
    )
    searcher_type = import_module(f"worldmem.retrieval.{preset['profile']}").Searcher
    memory._searcher = searcher_type(
        config, llm_client=llm, embedding_client=embeddings, **memory.retrieval_options
    )
    previous_usage = metadata.get("token_usage", {}) if resume else {}
    settings = dict(
        profile=preset["profile"],
        llm_model=preset["llm"].get("resolved_model", preset["llm"].get("model")),
        llm_backend=preset["llm"].get("resolved_backend", getattr(llm, "backend", "unknown")),
        embedding_model=preset["embedding"].get("model"),
        retrieval=memory.retrieval_options,
    )
    run_metadata = dict(
        fingerprint=signature,
        profile=preset["profile"],
        experiment=experiment,
        settings=settings,
        memory_hashes=memory_hashes,
        token_count="server-reported when present; otherwise estimated",
    )

    def run(job):
        key, group, owner, qa = job
        answer = memory.ask(owner, qa["question"], experiment=experiment)
        record = dict(qa, owner_id=owner, response=answer["response"], _worldmem_id=key)
        if full_results:
            record["retrieval"] = answer
        return (key, group, record)

    write_json(metadata_path, dict(run_metadata, done=sorted(done), token_usage=previous_usage))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            key, group, record = future.result()
            results.setdefault(group, []).append(record)
            done.add(key)
            write_json(output, results)
            current = memory._searcher._token_stats[experiment]
            usage = {k: previous_usage.get(k, 0) + v for k, v in current.items()}
            write_json(
                metadata_path,
                dict(run_metadata, done=sorted(done), token_usage=usage),
            )
    return results
