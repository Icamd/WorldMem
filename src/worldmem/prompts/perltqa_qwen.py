from jinja2 import Template

from .common import (
    DEFAULT_SEMANTIC_LABELS as DEFAULT_SEMANTIC_LABELS,
)
from .common import (
    FINAL_ANSWER_PROMPT as FINAL_ANSWER_PROMPT,
)
from .common import (
    INFERENCE_PATTERNS as INFERENCE_PATTERNS,
)
from .common import (
    MENTAL_KEYWORD_SELECTION_PROMPT as MENTAL_KEYWORD_SELECTION_PROMPT,
)
from .common import (
    MENTAL_QUERY_ANALYSIS_PROMPT as MENTAL_QUERY_ANALYSIS_PROMPT,
)
from .common import (
    NO_MENTAL_SELECTION_PROMPT as NO_MENTAL_SELECTION_PROMPT,
)
from .common import (
    TIME_FINAL_ANSWER_PROMPT as TIME_FINAL_ANSWER_PROMPT,
)
from .common import (
    TRIMEM_INFERENCE_PROMPT as TRIMEM_INFERENCE_PROMPT,
)

SITUATION_DECISION_PROMPT = Template("""
You are answering a question using retrieved situation evidence and semantic evidence from the memory bank.

This is a decision stage:
- If the current evidence is already sufficient, you MUST output the final answer directly.
- If the current evidence is not sufficient, you MUST say the answer is not yet determined and list the semantic contents that should be expanded further.

Requirements:
1. Read all provided evidence carefully.
2. Decide whether the answer is already determined from the current evidence.
3. If determined:
   - output the final answer directly
   - the answer should be concise, ideally under 5-10 words
4. If not determined:
   - leave answer as empty string
   - list all semantic contents that should be followed up on to find further evidence
5. For \"supporting_semantic_contents\" and \"need_expand_semantic_contents\", return the semantic text strings themselves, NOT numeric indices.
6. Return JSON only.

Question:
{{ question }}

Situation evidence:
{{ situation_evidence }}

Situation model evidence:
{{ situation_model_evidence }}

Return JSON only in this format:
{
  \"answer_determined\": true,
  \"proposed_answer\": \"....\",
  \"supporting_semantic_contents\": [\"...\", \"...\"],
  \"need_expand_semantic_contents\": []
}
or
{
  \"answer_determined\": false,
  \"proposed_answer\": \"\",
  \"supporting_semantic_contents\": [],
  \"need_expand_semantic_contents\": [\"...\", \"...\"]
}
""")
TRIMEM_FACTUAL_PROMPT = Template("""
Answer the question using ONLY the provided semantic-message pairs.

Question: {{ question }}

[Semantic-Message Pairs]
{{ semantic_message_pairs }}

CRITICAL RULES:
1. Answer concisely with the key facts.
2. Do NOT add unnecessary elaboration.
3. Answer must be based ONLY on the provided evidence. If not found, say \"Not mentioned in the conversation\".
4. Pay attention to WHO said/did things.
5. Resolve relative time expressions when needed.
6. Format dates as 'DD Month YYYY' when exact date is inferable.
7. For \"which country/state\" questions, answer with the country/state, NOT a city.
8. Return JSON only.

Examples:
Q: \"What career is she pursuing?\" → {\"reasoning\": \"Context says she chose counseling\", \"answer\": \"counseling\"}
Q: \"What does she do to destress?\" → {\"reasoning\": \"Context mentions running and pottery\", \"answer\": \"Running, pottery\"}
Q: \"What did she paint?\" → {\"reasoning\": \"Context says she painted a sunset\", \"answer\": \"sunset\"}
Q: \"Where has she camped?\" → {\"reasoning\": \"Mentions beach, mountains, forest\", \"answer\": \"beach, mountains, forest\"}
Q: \"Which country did he visit?\" → {\"reasoning\": \"Context says he went to Paris\", \"answer\": \"France\"}

Return ONLY valid JSON:
{\"reasoning\": \"...\", \"answer\": \"concise factual answer\"}
""")
PERLTQA_CATEGORY_ALIASES = {
    "profile": "Profile",
    "profiles": "Profile",
    "social relationship": "Social Relationship",
    "social relationships": "Social Relationship",
    "social_relationship": "Social Relationship",
    "social-relationship": "Social Relationship",
    "relationship": "Social Relationship",
    "relationships": "Social Relationship",
    "events": "Events",
    "event": "Events",
    "dialogues": "Dialogues",
    "dialogue": "Dialogues",
    "dialogs": "Dialogues",
    "dialog": "Dialogues",
}
PERLTQA_CATEGORY_ORDER = ["Profile", "Social Relationship", "Events", "Dialogues"]
PERLTQA_TIME_FINAL_ANSWER_PROMPT = Template("""
You are answering a PerLTQA question using structured memories and their source context.

Your job is to answer ONLY the time/date/duration asked by the question.

Important rules:
1. Evidence is grouped by source type: profile, social relationship, and events/dialogues.
2. Only events/dialogues evidence normally has reliable timestamps. Use timestamps from events/dialogues when answering time questions.
3. If the event text or dialogue contains a relative time expression, resolve it using the attached timestamp or event creation time when possible.
4. Do not answer with a timestamp merely because it appears in the evidence; answer the time of the target event/fact asked by the question.
5. Return only the concise answer text. No explanation.

Question:
{{ question }}

[PerLTQA Evidence]
{{ evidence_text }}

Answer:
""")
PERLTQA_FACTUAL_FINAL_ANSWER_PROMPT = Template("""
Answer the PerLTQA question using ONLY the provided structured memory evidence.

Question:
{{ question }}

[PerLTQA Evidence]
{{ evidence_text }}

Rules:
1. Use the evidence jointly across all sections.
2. Do not hallucinate facts not supported by evidence.
3. Pay attention to who each memory is about and who spoke in the dialogue context.
4. If evidence includes an event summary/content and dialogue block, prefer direct dialogue when it answers the question, but use the event content for broader context.
5. Answer concisely, ideally in one short sentence or phrase.
6. Return JSON only in this format:
{\"reasoning\": \"brief evidence-based reasoning\", \"answer\": \"concise answer\"}
""")
PERLTQA_SITUATION_DECISION_WITH_CATEGORY_PROMPT = Template("""
You are using retrieved PerLTQA semantic memories to decide which evidence should be expanded before final answering.

The semantic memories are grouped by their memory source:
- profile: stable personal attributes of the current character.
- social_relationship: relationships between the current character and supporting characters.
- events: event/dialogue memories, including activities, experiences, conversations, decisions, time, place, and detailed contextual facts.

Task:
1. Decide which category the current question is mainly asking about.
   Choose exactly one of: \"profile\", \"social_relationship\", \"events\".
   - Choose \"profile\" for gender, age, nickname, title, occupation, nationality, appearance, hobbies, achievements, education, employer, identity-like attributes.
   - Choose \"social_relationship\" for questions asking who someone is, how two people are connected, family/friend/colleague/partner relationships, or the name of a related person.
   - Choose \"events\" for questions about activities, conversations, experiences, opinions, decisions, guidance, causes, outcomes, dates, places, processes, and anything grounded in events/dialogues.
2. Decide whether the provided semantic evidence is already enough to answer the question.
3. Select all the semantic evidence that supports the answer.
4. Select all semantic evidence that should be expanded with source messages / event memories before final answering.

Important output rules:
- Return JSON only.
- For selected semantic evidence, return BOTH integer indices and exact semantic texts when possible.
- Indices must come from the \"Index <n>\" labels in the evidence below.
- Do not invent evidence.
- It is better to include several relevant semantics than to omit a detail needed for a complete PerLTQA-style answer.

Question:
{{ question }}

[Classified Semantic Evidence]
{{ classified_semantic_evidence }}

Return JSON only in this format:
{
  \"query_memory_category\": \"profile | social_relationship | events\",
  \"answer_determined\": true,
  \"proposed_answer\": \"a draft full-sentence answer if already clear, otherwise empty string\",
  \"supporting_semantic_indices\": [0, 3],
  \"supporting_semantic_contents\": [\"exact semantic text\", \"exact semantic text\"],
  \"need_expand_semantic_indices\": [5],
  \"need_expand_semantic_contents\": [\"exact semantic text\"]
}
""")
PERLTQA_UNIFIED_FINAL_ANSWER_PROMPT = Template("""
You are answering a PerLTQA question using ONLY the provided structured memory evidence and source context.

PerLTQA answers should be complete declarative sentences, but they should NOT be verbose. Your goal is to match the reference-answer style: complete, direct, and limited to the exact fact asked by the question.

Question:
{{ question }}

Predicted question memory category: {{ query_memory_category }}

PerLTQA Evidence
{{ evidence_text }}

Answer style requirements:
1. Return a complete declarative sentence, not a bare phrase.
2. Return JSON only in this format:
{\"reasoning\": \"brief evidence-based reasoning\", \"answer\": \"concise complete PerLTQA-style answer\"}
""")
