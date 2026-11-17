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

Select ALL evidence that could have relation with the question, so that we can further expand the source message based on these evidence. Ensure there are no omissions, and provide at least {{ min_semantics }} pieces of evidence.

Question:
{{ question }}

[Semantic evidence]
{{ semantic_evidence }}

Return JSON only in this format:
{
  \"supporting_semantic_contents\": [\"...\", \"...\"]
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
3. Answer must be based ONLY on the provided evidence.
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
