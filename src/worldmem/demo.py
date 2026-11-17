"""Synthetic integration fixture. It verifies plumbing, not model quality."""

import json
from contextlib import nullcontext
from pathlib import Path

from .api import WorldMem
from .clients import EmbeddingClient, LLMResponse, parse_json
from .config import MemoryConfig
from .runner import build_dataset
from .storage import write_json

FACT = "Alice visited the museum in Paris on 7 May 2023."
QUESTION = "Which city did Alice visit the museum in?"


class SyntheticLLM:
    """Scripted responses for one fictional museum example; no external requests."""

    backend = "synthetic"
    model_name = model = "synthetic-museum-fixture"
    tokenizer = None

    def __init__(self):
        self.calls = []

    def api_call_scope(self, **kwargs):
        return nullcontext()

    def generate_text_response_with_usage(self, prompt, **kwargs):
        self.calls.append(prompt)
        if "fixed inventory of world-memory labels" in prompt:
            result = dict(
                query_label_keywords={"Character": ["Alice"], "Event": ["museum"]},
                expected_answer_labels=["Location"],
            )
        elif "most answer-relevant candidate keywords" in prompt:
            result = dict(selected_keywords={"Location": ["Paris"], "Event": ["museum"]})
        elif (
            "Situation evidence" in prompt
            or "decision stage" in prompt
            or "decide which evidence" in prompt
            or ("supporting_semantic_contents" in prompt)
        ):
            result = dict(
                answer_determined=False,
                proposed_answer="",
                supporting_semantic_contents=[FACT],
                need_expand_semantic_contents=[FACT],
                query_memory_category="events",
            )
        elif "valuable FACTUAL information" in prompt:
            result = [
                dict(
                    lossless_restatement=FACT,
                    keywords=["Alice", "Paris", "museum"],
                    timestamp="2023-05-07T00:00:00",
                    persons=["Alice"],
                    entities=["museum"],
                    topic="museum visit",
                    source_message_indices=[0],
                )
            ]
        elif '"existing_labels"' in prompt and '"new_labels"' in prompt:
            result = dict(
                existing_labels={
                    "Character": ["Alice"],
                    "Location": ["Paris"],
                    "Event": ["museum"],
                },
                new_labels={},
            )
        elif prompt.lstrip().startswith("You are an episodic memory generation expert"):
            result = dict(title="Museum visit", content=FACT, timestamp="2023-05-08T13:56:00")
        elif "breakpoints" in prompt:
            result = dict(breakpoints=[], reasoning="one episode")
        elif "statements" in prompt:
            result = dict(statements=[FACT])
        elif "selected_indices" in prompt:
            result = dict(selected_indices=[0])
        else:
            if "Paris" not in prompt:
                raise AssertionError(
                    "The synthetic final-answer fixture requires source evidence containing Paris"
                )
            result = dict(reasoning="The fictional source explicitly says Paris.", answer="Paris")
        content = json.dumps(result)
        usage = dict(
            prompt_tokens_est=len(prompt) // 4,
            completion_tokens_est=len(content) // 4,
            total_tokens_est=(len(prompt) + len(content)) // 4,
        )
        return LLMResponse(content, usage, self.model_name)

    def generate_text_response(self, prompt, **kwargs):
        return self.generate_text_response_with_usage(prompt, **kwargs).content

    generate_response = generate_text_response

    def generate_json_response_with_usage(self, prompt, **kwargs):
        response = self.generate_text_response_with_usage(prompt, **kwargs)
        return (parse_json(response.content), response)

    def generate_json_response(self, prompt, **kwargs):
        return self.generate_json_response_with_usage(prompt, **kwargs)[0]

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def close(self):
        pass


def synthetic_locomo():
    return [
        dict(
            conversation=dict(
                speaker_a="Alice",
                speaker_b="Bob",
                session_1_date_time="1:56 pm on 8 May, 2023",
                session_1=[
                    dict(
                        speaker="Alice",
                        text="I visited the museum in Paris yesterday.",
                        dia_id="D1:1",
                    ),
                    dict(
                        speaker="Bob",
                        text="You visited the museum in Paris on May 7.",
                        dia_id="D1:2",
                    ),
                ],
            ),
            qa=[dict(question=QUESTION, answer="Paris", category=4)],
        )
    ]


def synthetic_perltqa():
    return {
        "Alice": dict(
            profile={"Protagonist": "Alice", "Occupation": "Astronomer"},
            profile_description="Alice is an astronomer. Alice enjoys museums.",
            social_relationship={
                "r1": {
                    "Supporting Characters": "Bob",
                    "Relationship": "friend",
                    "Description": "Alice's friend Bob",
                }
            },
            events={
                "e1": {"summary": "Museum visit", "content": FACT, "Creation Time": "2023-05-07"}
            },
            dialogues={
                "d1": {
                    "events": "e1",
                    "contents": {
                        "2023-05-08T13:56:00": [
                            "Alice: I visited the museum in Paris yesterday.",
                            "Bob: You visited the museum in Paris on May 7.",
                        ]
                    },
                }
            },
        )
    }


def run_demo(directory):
    directory = Path(directory)
    llm, embeddings = (SyntheticLLM(), EmbeddingClient(backend="hash"))
    memory_path = directory / "memories"
    preset = dict(
        profile="locomo_qwen",
        dataset="locomo",
        memory=dict(storage_path=str(memory_path)),
        llm=dict(model=llm.model_name),
        embedding=dict(backend="hash"),
        construction=dict(batch_size=40),
    )
    counts = build_dataset(synthetic_locomo(), preset, llm, embeddings)
    memory = WorldMem(
        MemoryConfig(storage_path=str(memory_path)), llm_client=llm, embedding_client=embeddings
    )
    result = memory.ask("Alice_0", QUESTION)
    write_json(directory / "answer.json", result)
    return dict(
        mode="synthetic; no model-quality claim",
        memory=counts,
        answer=result["response"],
        artifact=str(directory / "answer.json"),
    )
