# Evaluation protocols

Install `.[eval]` for offline metrics. Benchmark answers are stored as a
dictionary of conversation/owner IDs to QA records with `question`, `answer`,
`response`, and `category` fields.

## Lightweight scorer

```bash
python -m worldmem score --input outputs/locomo_api/answers.full.json \
  --output outputs/locomo_api/metrics.json
```

This preserves the selected repository's lightweight metric definitions:
case-insensitive exact match, F1 over **sets** of regex-tokenized words,
NLTK sentence BLEU with method-1 smoothing, and tokenized longest-common-subsequence
ROUGE-L. Set F1 differs from frequency-based token F1; do not compare it to another
implementation without checking tokenization and normalization.

## PerLTQA TriMem-style protocol

The original `evaluate_perltqa_trimem_metrics.py` is retained as
`worldmem.evaluation.perltqa_trimem`. Its category-specific judge prompt,
normalization, BLEU, and stemmed `rouge_score` definitions are separate from the
lightweight scorer. The optional judge evaluates predicted answers against
reference answers after retrieval; it is not part of memory construction or
answer generation.

Use an output name following `<prefix>.<experiment>.json`:

```bash
python -m worldmem.evaluation.perltqa_trimem \
  --results_dir outputs/perltqa_api --result-prefix answers --experiments full \
  --no-llm-judge --max_workers 1
```

All four PerLTQA categories are evaluated by default. To reproduce a narrower
selection, pass `--categories profile events`, for example. `--per-category-limit`
and `--max-questions` are trial-run limits, not full-benchmark settings.

LLM judging is off by default. To enable it explicitly:

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_API_KEY=EMPTY
python -m worldmem.evaluation.perltqa_trimem \
  --results_dir outputs/perltqa_api --result-prefix answers --experiments full \
  --llm-judge --judge-model YOUR_SERVED_JUDGE_MODEL --max_workers 1
```

For an external judge, configure its endpoint, model, and credentials through
the same environment variables. Judging may incur API charges when an external
endpoint is selected. The preserved evaluator uses Chat Completions with
sampling temperature and does not apply the GPT-5 reasoning policy used by the
answering client; select a judge endpoint compatible with that request format.
Document the exact judge model and settings when reporting scores.

No NLTK data is downloaded automatically. Word tokenization uses NLTK's
`preserve_line=True` mode, with a regex fallback. Metric differences between this
protocol and the lightweight scorer are intentional and should be reported.

## Token accounting

`answers.full.run.json` stores the model/embedding IDs, effective retrieval
settings, memory-file checksums, accumulated retrieval input/output tokens, and
query counts. Reasoning tokens are included in server-reported completion
usage. Construction and judge calls are not included in retrieval Token/Query.
HF generation uses its tokenizer; endpoints without usage metadata use the
research implementation's estimates. Synthetic-demo token counts are estimates
and are not benchmark measurements.
