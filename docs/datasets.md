# Dataset adapters and storage

The JSON files in `examples/` contain fictional data and illustrate the accepted
formats. They are not subsets of the evaluation benchmarks.

## LoCoMo

The input is a list. Each record has `conversation` and `qa` fields.
`conversation` contains `speaker_a`, `speaker_b`, `session_N` lists, and their
`session_N_date_time` values. Messages use `speaker` and `text`; image captions
and search queries are preserved when present. Sessions retain their dataset
timestamps so segmentation cannot accidentally merge the entire history into
one block when a model returns no breakpoints.

QA records use `question`, `answer`, and `category`. Categories 1–4 correspond
to the selected scripts' evaluated question groups. The preset excludes category
5; another selection can be passed using `--categories`.

```bash
python -m worldmem build --config configs/locomo_api.json \
  --data examples/locomo.synthetic.json --memory-dir outputs/example-locomo
python -m worldmem retrieve --config configs/locomo_api.json \
  --data examples/locomo.synthetic.json --memory-dir outputs/example-locomo \
  --output outputs/example-locomo/answers.full.json
```

These commands use a real API. Use `worldmem demo` for a fully offline run.

## PerLTQA

Memory and questions are separate files in English v2.

The memory adapter accepts a character-keyed dictionary, a list of character
objects, or one character object. A character contains:

- `profile`: structured attributes, including `Protagonist`.
- `profile_description`: the original descriptive text.
- `social_relationship`: relation-ID-keyed objects with `Supporting Characters`,
  `Relationship`, and `Description`.
- `events`: event-ID-keyed objects with lowercase `content` and optional
  `summary`, `Creation Time`, `Characters`, and `Theme` fields.
- `dialogues`: dialogue-ID-keyed objects with an `events` pointer and `contents`,
  mapping timestamps to speaker-prefixed utterances.

Profiles become field-level semantic entries. Social descriptions preserve their
original source block. Event content is split into source sentences; dialogues
are split into utterances. Atomic extraction runs over event/dialogue blocks,
retaining `source`, `source_subtype`, event IDs, dialogue IDs, and message links.
The historic `events` metadata source also covers dialogue utterances; the
subtype distinguishes them. No QA answer or reference-memory field is used to
construct memory.

The QA adapter accepts a character-keyed dictionary or a list of such dictionaries.
Each character has `profile`, `social_relationship`, `events`, and `dialogues`
question groups. Profile questions can be a direct list; the other groups
typically use a source-ID-keyed list. QA entries use `Question`, `Answer`, and
optional `Reference Memory` / `Memory Anchors` fields. Those reference fields
are copied to results after answering and never enter model prompts.

## Canonical files

```text
MEMORY_DIR/
  message_blocks/OWNER_message_blocks.jsonl
  episodes/OWNER_episodes.jsonl
  semantic/OWNER_semantic.jsonl
  situation/OWNER_situation.jsonl
  mental/OWNER_mental.jsonl
  checkpoints/OWNER.json
  owners.json
```

Memory record fields remain compatible with the selected implementation's JSONL
schema. Semantic records retain `source_episodes`, `matched_message_id`,
`matched_message_index`, `matched_block_id`, `source_message_ids`, and label
annotations. Situation aggregates reference semantic IDs; Mental units group
labels and link their semantic entries and situations.

Generated UUIDs and record creation timestamps are not stable across rebuilds.
The input/model fingerprint and output-file checksums determine resume status.
A successful build requires every memory record type to be nonempty.
The CLI rebuilds one owner in a staging directory, then promotes its files and
writes a completion checkpoint. A process interruption during promotion is
detected by checksum validation on the next run. Use one build process per
memory directory.

Existing JSONL memories can be queried using `--memory-dir` if their owner IDs
and layer names match. They have no new build checkpoint, so reconstruction into
the same directory requires `--overwrite`. A new directory is preferable when
changing the construction model.
