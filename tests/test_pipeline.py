import json
from pathlib import Path

import pytest

from worldmem import MemoryConfig, WorldMem
from worldmem.clients import EmbeddingClient
from worldmem.demo import FACT, QUESTION, SyntheticLLM, synthetic_locomo, synthetic_perltqa
from worldmem.runner import build_dataset, retrieve_dataset
from worldmem.storage import LAYERS, layer_path, read_jsonl


@pytest.fixture
def clients():
    return SyntheticLLM(), EmbeddingClient(backend="hash")


def preset_for(profile, tmp_path):
    preset = json.loads((Path(__file__).parents[1] / "configs" / f"{profile}.json").read_text())
    preset["memory"]["storage_path"] = str(tmp_path / "memories")
    preset["embedding"]["backend"] = "hash"
    return preset


@pytest.mark.parametrize("profile", ["locomo_qwen", "locomo_api", "perltqa_qwen", "perltqa_api"])
def test_full_construction_retrieval_and_provenance(profile, tmp_path, clients):
    llm, embedding = clients
    preset = preset_for(profile, tmp_path)
    dataset = synthetic_locomo() if profile.startswith("locomo") else synthetic_perltqa()
    summary = build_dataset(dataset, preset, llm, embedding)
    owner = summary[0]["owner"]
    assert summary[0]["status"] == "built"
    layers = {
        name: read_jsonl(layer_path(preset["memory"]["storage_path"], owner, name))
        for name in LAYERS
    }
    assert all(layers.values())
    source_ids = {
        message["message_id"] for block in layers["message_blocks"] for message in block["messages"]
    }
    semantics = {item["memory_id"]: item for item in layers["semantic"]}
    assert any(item["content"] == FACT for item in semantics.values())
    if profile.startswith("perltqa"):
        assert summary[0]["counts"]["accepted_dialogue_atomic_semantics"] >= 2
        assert {"profile", "social_relationship", "events"} <= {
            item.get("metadata", {}).get("source") for item in semantics.values()
        }
    assert source_ids
    assert all(
        item.get("matched_message_id") in source_ids
        for item in semantics.values()
        if item.get("matched_message_id")
    )
    for situation in layers["situation"]:
        for aggregate in situation["aggregates"]:
            assert set(aggregate["semantic_ids"]) <= semantics.keys()
    memory = WorldMem(
        MemoryConfig.from_dict(preset["memory"]),
        llm_client=llm,
        embedding_client=embedding,
        profile=profile,
        retrieval_options=preset["retrieval"],
    )
    result = memory.ask(owner, QUESTION)
    assert result["response"] == "Paris"
    assert result["mental"]
    assert result["situation"]
    assert "visited the museum" in json.dumps(result)
    calls = len(llm.calls)
    resumed = build_dataset(dataset, preset, llm, embedding)
    assert resumed[0]["status"] == "skipped"
    assert len(llm.calls) == calls


@pytest.mark.parametrize("experiment", ["full", "no_mental", "no_situation"])
def test_ablations_are_executable(experiment, tmp_path, clients):
    llm, embeddings = clients
    preset = preset_for("locomo_qwen", tmp_path)
    build_dataset(synthetic_locomo(), preset, llm, embeddings)
    memory = WorldMem(
        MemoryConfig.from_dict(preset["memory"]),
        llm_client=llm,
        embedding_client=embeddings,
        retrieval_options=preset["retrieval"],
    )
    result = memory.ask("Alice_0", QUESTION, experiment=experiment)
    assert result["response"] == "Paris"


def test_resume_fingerprints_and_staged_failure_preserve_committed_memory(tmp_path, clients):
    llm, embeddings = clients
    preset = preset_for("locomo_qwen", tmp_path)
    build_dataset(synthetic_locomo(), preset, llm, embeddings)
    root = Path(preset["memory"]["storage_path"])
    before = {name: layer_path(root, "Alice_0", name).read_bytes() for name in LAYERS}
    changed = synthetic_locomo()
    changed[0]["conversation"]["session_1"][0]["text"] = "A changed interaction."
    with pytest.raises(ValueError, match="changed"):
        build_dataset(changed, preset, llm, embeddings)

    class BrokenLLM(SyntheticLLM):
        def generate_json_response(self, *args, **kwargs):
            raise RuntimeError("simulated endpoint outage")

    with pytest.raises(RuntimeError):
        build_dataset(changed, preset, BrokenLLM(), embeddings, overwrite=True)
    assert before == {name: layer_path(root, "Alice_0", name).read_bytes() for name in LAYERS}
    assert not list((root / ".staging").iterdir())


def test_benchmark_reference_is_never_sent_to_the_model_and_resume_skips_calls(tmp_path, clients):
    llm, embeddings = clients
    preset = preset_for("locomo_qwen", tmp_path)
    data = synthetic_locomo()
    build_dataset(data, preset, llm, embeddings)
    llm.calls.clear()
    data[0]["qa"][0]["answer"] = "REFERENCE_ANSWER_MUST_NOT_ENTER_MODEL_CONTEXT"
    output = tmp_path / "results.json"
    result = retrieve_dataset(data, preset, llm, embeddings, output=output, full_results=True)
    assert result["0"][0]["response"] == "Paris"
    assert all(
        "REFERENCE_ANSWER_MUST_NOT_ENTER_MODEL_CONTEXT" not in prompt for prompt in llm.calls
    )
    call_count = len(llm.calls)
    retrieve_dataset(data, preset, llm, embeddings, output=output, full_results=True)
    assert len(llm.calls) == call_count
    build_dataset(data, preset, llm, embeddings, overwrite=True)
    with pytest.raises(ValueError, match="memory"):
        retrieve_dataset(data, preset, llm, embeddings, output=output, full_results=True)


def test_perltqa_memory_and_qa_files_resolve_the_same_owner(tmp_path, clients):
    llm, embeddings = clients
    preset = preset_for("perltqa_qwen", tmp_path)
    memories = {"numeric-id": synthetic_perltqa()["Alice"]}
    summary = build_dataset(memories, preset, llm, embeddings)
    qa = {"Alice": {"events": [{"e1": [{"Question": QUESTION, "Answer": "Paris"}]}]}}
    result = retrieve_dataset(qa, preset, llm, embeddings, output=tmp_path / "answers.json")
    assert next(iter(result)) == summary[0]["owner"]
    assert next(iter(result.values()))[0]["response"] == "Paris"


def test_missing_owner_is_an_error_instead_of_an_ungrounded_answer(tmp_path, clients):
    llm, embeddings = clients
    memory = WorldMem(
        MemoryConfig(storage_path=str(tmp_path)), llm_client=llm, embedding_client=embeddings
    )
    with pytest.raises(FileNotFoundError):
        memory.ask("missing", QUESTION)
    with pytest.raises(ValueError):
        memory.ask("../escape", QUESTION)
    assert not llm.calls


def test_parallel_episode_failure_does_not_commit_a_partial_memory(tmp_path):
    class PartialFailureLLM(SyntheticLLM):
        def generate_json_response(self, prompt, **kwargs):
            if prompt.lstrip().startswith("You are an expert at segmenting"):
                return {"breakpoints": [0, 1], "reasoning": "two episodes"}
            if prompt.lstrip().startswith("You are an episodic") and "FAIL_THIS_EPISODE" in prompt:
                raise RuntimeError("simulated one-episode failure")
            return super().generate_json_response(prompt, **kwargs)

    preset = preset_for("locomo_qwen", tmp_path)
    preset["memory"]["max_workers"] = 2
    data = synthetic_locomo()
    data[0]["conversation"]["session_1"][1]["text"] = "FAIL_THIS_EPISODE"
    with pytest.raises(RuntimeError, match="one-episode failure"):
        build_dataset(data, preset, PartialFailureLLM(), EmbeddingClient(backend="hash"))
    root = Path(preset["memory"]["storage_path"])
    assert not (root / "checkpoints/Alice_0.json").exists()
    assert not layer_path(root, "Alice_0", "episodes").exists()
