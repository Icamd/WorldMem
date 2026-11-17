import json

import httpx
import pytest
from openai import OpenAI

from worldmem.clients import LLMClient, parse_json
from worldmem.demo import SyntheticLLM


def connected_client(backend):
    requests = []
    scripted = SyntheticLLM()

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        output = scripted.generate_text_response_with_usage(body["messages"][-1]["content"])
        return httpx.Response(
            200,
            json=dict(
                id="test-completion",
                object="chat.completion",
                created=0,
                model=body["model"],
                choices=[
                    dict(
                        index=0,
                        message=dict(role="assistant", content=output.content),
                        finish_reason="stop",
                    )
                ],
                usage=dict(
                    prompt_tokens=25,
                    completion_tokens=20,
                    total_tokens=45,
                    completion_tokens_details=dict(reasoning_tokens=7),
                ),
            ),
        )

    client = LLMClient(
        backend=backend,
        model="gpt-5-nano" if backend == "openai_api" else "Qwen3-8B",
        api_key="test-only",
        base_url="http://localhost:9999/v1",
    )
    client.client.close()
    client.client = OpenAI(
        api_key="test-only",
        base_url="http://localhost:9999/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    return client, requests


def test_gpt_reasoning_budgets_and_usage_are_transmitted_correctly():
    client, requests = connected_client("openai_api")
    with client.api_call_scope(reasoning_effort="medium", min_completion_tokens=16384):
        result = client.generate_text_response_with_usage(
            "test Paris", max_tokens=100, temperature=0.7
        )
    body = requests[0]
    assert body["max_completion_tokens"] == 16384
    assert body["reasoning_effort"] == "medium"
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert "chat_template_kwargs" not in body
    assert result.usage["thinking_tokens_est"] == 7
    assert result.usage["total_tokens_est"] == 45
    client.generate_text_response("test Paris", max_tokens=5000)
    assert requests[-1]["reasoning_effort"] == "minimal"
    client.close()


def test_vllm_sampling_and_qwen_thinking_are_sent_as_extensions():
    client, requests = connected_client("vllm")
    client.generate_text_response(
        "test Paris", max_tokens=123, enable_thinking=True, top_k=20, top_p=0.95
    )
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert requests[0]["top_k"] == 20
    assert requests[0]["max_tokens"] == 123
    assert "reasoning_effort" not in requests[0]
    client.close()


@pytest.mark.parametrize(
    "text", ['<think>reasoning</think>```json\n{"x": [1]}\n```', 'prefix {"x": [1]} trailing']
)
def test_structured_output_parser(text):
    assert parse_json(text) == {"x": [1]}


def test_invalid_json_is_not_reported_as_an_empty_success():
    with pytest.raises(ValueError):
        parse_json("not JSON")


def test_vllm_does_not_reuse_external_gpt_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-fixture")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    client = LLMClient(backend="vllm", model="fixture", base_url="http://localhost:9999/v1")
    assert client.client.api_key == "EMPTY"
    client.close()


@pytest.mark.parametrize("profile", ["locomo_api", "perltqa_api"])
def test_api_profiles_run_through_the_real_sdk_transport(profile, tmp_path):
    from pathlib import Path

    from worldmem import MemoryConfig, WorldMem
    from worldmem.clients import EmbeddingClient
    from worldmem.demo import QUESTION, synthetic_locomo, synthetic_perltqa
    from worldmem.runner import build_dataset

    preset = json.loads((Path(__file__).parents[1] / "configs" / f"{profile}.json").read_text())
    preset["memory"]["storage_path"] = str(tmp_path / "memories")
    client, requests = connected_client("openai_api")
    embeddings = EmbeddingClient(backend="hash")
    dataset = synthetic_locomo() if profile.startswith("locomo") else synthetic_perltqa()
    summary = build_dataset(dataset, preset, client, embeddings)
    memory = WorldMem(
        MemoryConfig.from_dict(preset["memory"]),
        llm_client=client,
        embedding_client=embeddings,
        profile=profile,
        retrieval_options=preset["retrieval"],
    )
    assert memory.ask(summary[0]["owner"], QUESTION)["response"] == "Paris"
    assert (
        requests[-1]["reasoning_effort"]
        == preset["retrieval"]["stage_reasoning_effort"]["situation_final_answer"]
    )
    assert (
        requests[-1]["max_completion_tokens"] >= preset["retrieval"]["openai_api_stage_max_tokens"]
    )
    assert all("temperature" not in body for body in requests)
    client.close()
