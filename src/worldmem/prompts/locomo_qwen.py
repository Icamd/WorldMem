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
3. Select all semantic evidence that supports the answer / should be expanded with source messages.
4. If determined:
   - output the final answer directly
   - select all the semantic that supports the answer
5. If not determined:
   - leave answer as empty string
   - select all semantic that supports the answer
   - list all semantic evidence that should be followed up on to find further evidence
6. For \"supporting_semantic_contents\" and \"need_expand_semantic_contents\", return the semantic text strings themselves, NOT numeric indices.
7. Return JSON only.

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
