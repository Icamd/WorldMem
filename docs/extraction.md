# Extraction and reproducibility notes

This release separates the active WorldMem implementation from a larger
experimental repository. It preserves the four selected retrieval variants and
their prompts; it does not include the older alternative versions.

## Entry-point mapping

Paths below are relative to the old WorldMem implementation directory. They are
provenance references, not runtime dependencies.

| Original entry | Independent implementation |
| --- | --- |
| `evaluation/locomo/add_wm_1.py` | `construction/builder.py`, `datasets/locomo.py`, `runner.build_dataset` |
| `evaluation/locomo/add_wm_1_external_api.py` | Same builder with the GPT API client and LoCoMo API preset |
| `evaluation/locomo/add_perltqa_wm.py` | `construction/perltqa.py`, `datasets/perltqa.py`, PerLTQA Qwen preset |
| `evaluation/locomo/add_perltqa_wm_gpt.py` | Same PerLTQA adapter with GPT stage policies and API preset |
| `evaluation/locomo/wMemory_search_framework_retrieval_2_thinking_sem_budget.py` | `retrieval/locomo_qwen.py` |
| `evaluation/locomo/wMemory_search_framework_retrieval_3_openai.py` | `retrieval/locomo_api.py` |
| `evaluation/locomo/wMemory_search_framework_retrieval_perltqa_v7_global_supplement.py` | `retrieval/perltqa_qwen.py` |
| `evaluation/locomo/wMemory_search_framework_retrieval_perltqa_gpt5_reasoning_v2.py` | `retrieval/perltqa_api.py` |
| `src/utils/llm_client_openllm.py` | Lazy optional Transformers implementation in `hf.py`, shared client in `clients.py` |
| `src/utils/llm_client_vllm_openai.py` | Shared client in `clients.py`; explicit server launcher in `scripts/serve_vllm.sh` |

The selected construction entry called `xMemory.build_world_memory`, which
delegated to `src/core/memory_system.py`; the similarly named
`world_memory_system.py` and `api/wmemory.py` were not the actual construction
path. PerLTQA explicitly selected `memory_system_perltqa.py`, whose differences
are retained in `PerLTQABuilder`.

## Model startup found in the executable code

The original LoCoMo local builder instantiated `LLMClient` with the default HF
backend, loading a tokenizer and causal model through `from_pretrained` in the
same process. Retrieval defaulted to a `VLLMOpenAIClient` connection to an
existing server, with an optional HF branch. The selected PerLTQA local builder
also defaulted to an existing vLLM server. External-API builders selected
`openai_api`, which did not load local LLM weights.

No automatic vLLM server startup appears in these entry points. The included
launcher makes that separate requirement explicit. Local Qwen presets now use
the requested Qwen model ID instead of the machine-specific paths and occasional
Llama defaults left in the old scripts.

## What was removed or changed

- Removed commented duplicate implementations, backup/version experiments,
  unrelated xMemory entry points, Theme graph updates, semantic kNN side files,
  Chroma/BM25 services, and LanceDB mirrors. The selected retrieval implementations
  read the canonical JSONL files, so those indices and mirrors are unnecessary.
- Kept episodic prediction/correction extraction, atomic backfilling,
  source-message alignment, keyword propagation, adaptive labels, situation
  assembly, query routing, global semantic supplementation, evidence selection,
  answer prompts, and the Mental/Situation ablations.
- Shared exactly matching retrieval methods through inheritance and placed
  experiment-specific prompts in separate modules. Prompt text and defaults
  originate from the selected executable variants.
- Replaced hardcoded device IDs, personal paths, gateways, and credentials with
  CLI options, portable model IDs, and environment variables. Importing the
  package does not load GPU libraries or model weights.
- Replaced the old service/repository graph with a JSONL repository. Removed
  unused model-prefix-cache and log-probability utilities that the selected
  WorldMem retrieval path did not call.
- Removed unused construction switches that never affected the final LoCoMo
  build. The active pipeline always performs message-block atomic extraction.
- Use staged builds, completion fingerprints, and checksums. Model/API failure
  must not be treated as a successful empty fallback episode. Preserve already
  committed memory when a staged rebuild fails.
- Keep PerLTQA label inventories separate from the LoCoMo inventory and resolve
  QA owner names through a build manifest. Strict storage reads surface malformed
  JSONL instead of silently interpreting it as absent evidence.
- Preserve rule-based metrics. The optional TriMem scorer reads credentials
  from the environment, disables LLM judging by default, selects all categories
  by default, and avoids automatic NLTK downloads.

The upstream MIT copyright notice is retained verbatim. It is third-party
attribution, not an identification of the anonymous submission's authors.

## Script defaults versus the paper

The supplied presets use the selected scripts' executable retrieval defaults.
Machine-specific paths and worker counts are not part of the algorithm. The
following differences should be considered before claiming an exact paper rerun:

| Setting | Paper appendix | LoCoMo Qwen | LoCoMo GPT | PerLTQA Qwen / GPT |
| --- | --- | --- | --- | --- |
| Mental labels considered | 40 | 30 | 50 | 50 |
| Extra Mental labels | 20 | 6 | 20 | 20 |
| Extra keywords per label | 10 | 6 | 10 | 10 |
| Situation model total | 50 | 50 | 50 | 40 |
| Global supplement size | 80 | 80 | 120 | 120 |
| Preserved Situation semantics | 30 | 20 | 30 | 30 |
| Situation candidate budget | Dataset/variant dependent | 85 | 120 | 120 |
| GPT final-answer effort | medium | n/a | medium | n/a / high |
| GPT retrieval stage budget | at least 16,384 | n/a | 16,384 | n/a / 32,768 |

The paper lists GPT construction minimum budgets of 16,384 for ordinary calls
and 32,768 for semantic extraction. The original LoCoMo core scopes requested
4,096 / 16,384 while the external client could impose its own configured minimum.
The LoCoMo API preset applies a 16,384 global floor; PerLTQA retains its 32,768
floor. To enforce the paper's 32,768 extraction floor specifically, adjust the
`memory.construction_semantic_min_tokens` to `32768` in the preset and record
that choice. `memory.construction_regular_min_tokens` controls the ordinary
construction floor. Local Qwen thinking is enabled
for message-block atomic extraction; local retrieval thinking defaults to off.

The release uses one worker by default for portable CPU clients. Construction
retains sequential owner processing; episode generation can use
`memory.max_workers`. Retrieval concurrency is controlled by `--workers`.
Random model generation, backend versions, floating-point kernels, and created
UUIDs can change outputs. The package provides executable source, not cached
benchmark answers or a claim of reproduced paper scores.

## Validation boundary

CPU-only integration tests cover the five record types, all four retrieval
profiles, both ablations, source IDs, PerLTQA's event/dialogue layout, owner
resolution, resume, staged failure recovery, absence of QA-reference leakage,
and GPT/vLLM request contracts. Scripted responses and hash embeddings are used
for these tests. GPU model loading and paid GPT calls must be checked separately
in the intended experimental environment.
