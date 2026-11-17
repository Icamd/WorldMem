"""Small public facade for construction and situation-mediated answering."""

from importlib import import_module

from .config import MemoryConfig
from .construction.builder import MemoryBuilder
from .storage import LAYERS, layer_path, validate_owner

PROFILES = ("locomo_qwen", "locomo_api", "perltqa_qwen", "perltqa_api")


def normalize_retrieval_options(options):
    options = dict(options or {})
    # The original entry scripts expanded this one switch into the three
    # concrete answer branches. Keep that convenience in the unified facade.
    for key in ("stage_thinking", "stage_reasoning_effort"):
        if key not in options:
            continue
        stages = dict(options[key])
        if "situation_final_answer" in stages:
            for name in (
                "situation_time_final_answer",
                "situation_inference_final_answer",
                "situation_factual_final_answer",
            ):
                stages.setdefault(name, stages["situation_final_answer"])
        options[key] = stages
    return options


class WorldMem:
    def __init__(
        self,
        config=None,
        *,
        llm_client,
        embedding_client,
        profile="locomo_qwen",
        retrieval_options=None,
    ):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        self.config = config or MemoryConfig()
        self.llm_client, self.embedding_client = (llm_client, embedding_client)
        self.profile = profile
        self.retrieval_options = normalize_retrieval_options(retrieval_options)
        self._searcher = None

    def build(self, owner_id, messages, *, batch_size=40):
        """Construct memory from messages in a new owner namespace."""
        validate_owner(owner_id)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if any(
            (
                layer_path(self.config.storage_path, owner_id, layer).exists()
                for layer in ("episodes", "semantic", "mental")
            )
        ):
            raise FileExistsError(
                "Owner already has memory; use the CLI's transactional build/resume workflow"
            )
        builder = MemoryBuilder(self.config, self.llm_client, self.embedding_client)
        result = builder.build_world_memory_from_messages(owner_id, messages, batch_size=batch_size)
        self._searcher = None
        return result

    def ask(self, owner_id, query, *, experiment="full"):
        """Return an answer plus the mental/situation evidence and retrieval trace."""
        validate_owner(owner_id)
        if not query.strip():
            raise ValueError("query must not be empty")
        for layer in LAYERS:
            path = layer_path(self.config.storage_path, owner_id, layer)
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"Missing or empty {layer} memory for owner {owner_id}")
        if self._searcher is None:
            searcher_type = import_module(f"worldmem.retrieval.{self.profile}").Searcher
            self._searcher = searcher_type(
                self.config,
                llm_client=self.llm_client,
                embedding_client=self.embedding_client,
                **self.retrieval_options,
            )
        searcher = self._searcher
        user_data = searcher._load_user_memories(owner_id)
        if experiment == "no_situation":
            mental = searcher._run_mental_full(query, user_data, exp_name="no_situation")
            result = searcher._run_no_situation_pipeline_for_query(query, user_data, mental)
        else:
            functions = {
                "full": searcher._run_full_pipeline_for_query,
                "no_mental": searcher._run_no_mental_pipeline_for_query,
            }
            if experiment not in functions:
                raise ValueError(f"Unknown experiment: {experiment}")
            result = functions[experiment](query, user_data)
        if not isinstance(result.get("response"), str) or not result["response"].strip():
            raise RuntimeError(
                "The answering pipeline returned an empty answer; this query was not completed"
            )
        return result

    def close(self):
        self.llm_client.close()
        self.embedding_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
