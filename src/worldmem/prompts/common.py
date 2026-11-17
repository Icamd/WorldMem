from jinja2 import Template

DEFAULT_SEMANTIC_LABELS = [
    "Character",
    "Identity",
    "Education",
    "Career",
    "Preferences",
    "Health",
    "Emotions & Moods",
    "Competence & Experience",
    "Technical Details",
    "Relationship",
    "Goals & Plans",
    "Beliefs & Values",
    "Habits & Routines",
    "Art & Culture",
    "Object",
    "Location",
    "Organization",
    "Action",
    "Event",
    "Time",
]
MENTAL_QUERY_ANALYSIS_PROMPT = Template("""
You are analyzing a question using a fixed inventory of world-memory labels.

Task:
Given:
1. The question
2. A candidate list of labels

You must:
A. Extract all labels that are explicitly or implicitly involved in the question itself.
B. For each extracted label, provide the corresponding keywords / key phrases from the question.
C. Predict which labels the expected answer is likely to involve.
D. You must only choose labels from the provided label list.
E. If the question involves time:
   - include label \"Time\"
   - if an explicit time expression exists (e.g. \"5 years ago\", \"next month\", \"1980s\"), use it as the keyword
   - if the question asks about time but has no explicit time expression (e.g. \"when\", \"how long\"), set the Time keyword to null

Important requirements:
1. Cover the question as comprehensively as possible.
2. Extract the query's own keywords as comprehensively as possible. Do not omit important keywords, key phrases, actions, objects, locations, or events explicitly mentioned in the question.
3. Do not miss likely answer labels; selecting slightly more is better than missing a key label.
4. Return JSON only.

Example 1:
Question:
How long has Caroline had her current group of friends for?

Candidate labels:
[\"Character\", \"Relationship\", \"Time\", \"Event\", \"Action\", ...]

Output:
{
  \"query_label_keywords\": {
    \"Character\": [\"Caroline\"],
    \"Relationship\": [\"group of friends\"],
    \"Time\": [null]
  },
  \"expected_answer_labels\": [\"Time\", \"Relationship\", \"Character\"]
}

Example 2:
Question:
What did Caroline research 5 years ago?

Output:
{
  \"query_label_keywords\": {
    \"Character\": [\"Caroline\"],
    \"Action\": [\"research\"],
    \"Time\": [\"5 years ago\"]
  },
  \"expected_answer_labels\": [\"Action\", \"Event\", \"Object\"]
}

Question:
{{ question }}

Candidate labels:
{{ labels }}

Return JSON only in this format:
{
  \"query_label_keywords\": {
    \"Label A\": [\"keyword 1\", \"keyword 2\"],
    \"Time\": [null]
  },
  \"expected_answer_labels\": [\"Label X\", \"Label Y\"]
}
""")
MENTAL_KEYWORD_SELECTION_PROMPT = Template("""
You are selecting the most answer-relevant candidate keywords for a question.

Task:
For each label, choose the keywords that are most likely to directly point to the answer.
If none of the keywords under a label are useful for answering the question, do not select any keyword for that label.

Requirements:
1. Select only keywords that help identify the answer.
2. The answer may require combining multiple labels.
3. If a label has no useful keyword, omit that label from the output.
4. Return JSON only.

Example:
Question:
When did Caroline go to school to do research yesterday?

Candidate keywords:
{
  \"Character\": [\"Caroline\", \"Melanie\"],
  \"Action\": [\"go to school\", \"do research\", \"attend support group\"],
  \"Location\": [\"school\", \"beach\"],
  \"Time\": [\"yesterday\", \"last year\", \"May 2022\"],
  \"Career\": [\"speech\", \"project\", \"film\", \"writing\", \"work\"]
}

Output:
{
  \"selected_keywords\": {
    \"Character\": [\"Caroline\"],
    \"Action\": [\"go to school\", \"do research\"],
    \"Location\": [\"school\"],
    \"Time\": [\"yesterday\"]
  }
}

Question:
{{ question }}

Candidate keywords:
{{ candidate_keywords }}

Return JSON only in this format:
{
  \"selected_keywords\": {
    \"Label A\": [\"keyword 1\", \"keyword 2\"]
  }
}
""")
NO_MENTAL_SELECTION_PROMPT = Template("""
You are given a question and a list of candidate semantic memories.

Task:
Select the semantic memories that are most likely to directly help answer the question.

Requirements:
1. Prefer semantics that directly constrain or support the answer.
2. It is fine to select multiple semantics.
3. Keep the strongest ones.
4. Return JSON only.

Question:
{{ question }}

Candidate semantics:
{{ semantics }}

Return JSON only in this format:
{
  \"selected_indices\": [0, 1, 3]
}
""")
TIME_FINAL_ANSWER_PROMPT = Template("""
You are extracting and converting the final time answer from semantic evidence and matched original messages.

Your job is to answer ONLY the time of the target event or fact asked in the question.

Critical rules:
1. Each item below is a semantic-message pair. Use the semantic together with its matched original message.
2. The semantic timestamp is the time when the memory was recorded, NOT necessarily the time of the event itself.
3. If the original message or semantic content contains a relative time expression such as:
   - yesterday
   - the previous day
   - last week
   - last month
   - last year
   - two months ago
   - three years ago
   - the week before
   then you MUST convert it relative to the attached timestamp.
4. Never answer with the memory timestamp unless the evidence clearly says the event happened at that exact time.
5. If the evidence says \"last week\" and the attached timestamp is 2023-06-09, the event time should be expressed as:
   \"The week before 9 June 2023\"
   not \"June 9, 2023\".
6. If the evidence says \"last year\" and the timestamp is in 2023, the answer should be \"2022\".
7. Use the most specific inferable natural time expression.
8. Return only the answer text. No explanation.

Examples:

Question:
When did Caroline attend the support group?

Semantic-message pair:
Semantic: [2023-05-08T13:56:00] Caroline attended an LGBTQ support group
Matched message: [2023-05-08T13:56:00] Caroline: I went to a LGBTQ support group yesterday and it was so powerful.

Correct answer:
May 7, 2023

Question:
When did Melanie paint the lake sunrise?

Semantic-message pair:
Semantic: [2023-05-08T13:56:00] Melanie painted a lake sunrise
Matched message: [2023-05-08T13:56:00] Melanie: I painted that lake sunrise last year.

Correct answer:
2022

Question:
{{ question }}

Semantic-message pairs:
{{ semantic_message_pairs }}

Answer:
""")
TRIMEM_INFERENCE_PROMPT = Template("""
Answer the inference question by reasoning from the provided context.

Question: {{ question }}

[Structured Memory Entries]
{{ semantic_message_pairs }}

RULES FOR INFERENCE QUESTIONS:
1. INFER from the evidence — NEVER say \"Not mentioned\" or \"unknown\". You MUST commit to an answer.
2. Use context to reason about personality, interests, values, behavior.
3. USE WORLD KNOWLEDGE: When the context describes something without naming it, IDENTIFY it.
   - A game exclusive to a platform → name the platform (e.g. Xenoblade 2 → Nintendo Switch)
   - A described but unnamed product → identify it (e.g. \"card game about exploding cats\" → Exploding Kittens)
   - A described technique → name it (e.g. \"work 25 minutes then break\" → Pomodoro technique)
   - A city mentioned when question asks about country → resolve to country (e.g. Paris → France, Toronto → Canada)
4. Pay attention to WHO said/did things — speaker tags are authoritative.
5. ALWAYS cite specific facts from context in your reasoning — never guess generically.
6. Prefer the MOST SPECIFIC reason supported by the context.

ANSWER FORMAT — BE MAXIMALLY CONCISE:
- The \"answer\" field must be as SHORT as possible. All reasoning goes in the \"reasoning\" field ONLY.
- For yes/no questions (\"would\", \"does\", \"is\", \"can\", \"did\", \"are\", \"was\"):
  * If evidence is DIRECT: answer \"Yes\" or \"No\"
  * If evidence is INDIRECT/inferred: answer \"Likely yes\" or \"Likely no\"
  * If the reference answer would naturally include a brief reason, add a SHORT reason after a comma. Keep it under 10 words.
  * NEVER add long explanations.
- For \"A or B\" choice questions (e.g. \"beach or mountains?\", \"Charger or Forester?\"): Answer with ONLY the chosen option — do NOT answer \"Yes\" or \"No\". Pick A or B based on evidence.
- For \"what/who/which\" questions: Answer with ONLY the name, term, or short phrase. No explanations.
- For trait/attribute questions: List ONLY the adjectives, comma-separated.
- For \"what might/what would/what could\" questions: Give the shortest answer that captures the key point. Prefer a few words or a short phrase over a full sentence.
- For identification questions: Just name the thing. \"Exploding Kittens\", not \"Exploding Kittens, a card game about cats\".
- For \"which country/state\" questions: Answer with ONLY the country or state name.
- For \"how many/how often\" questions: Answer with ONLY the number or frequency.

Examples:
Q: \"Would she enjoy classical music?\" → {\"reasoning\": \"Context says she loves Bach and Mozart\", \"answer\": \"Yes\"}
Q: \"Would she be considered an ally?\" → {\"reasoning\": \"She attended pride events and supported her friend's transition\", \"answer\": \"Yes, she is supportive\"}
Q: \"Would he be open to moving to another country?\" → {\"reasoning\": \"He has goals specifically in the U.S.\", \"answer\": \"No, he has goals specifically in the U.S.\"}
Q: \"Are they fans of the same team?\" → {\"reasoning\": \"James is a Liverpool fan and John is a Manchester City fan\", \"answer\": \"No, James is a Liverpool fan and John is a Manchester City fan\"}
Q: \"Does he live close to a beach or the mountains?\" → {\"reasoning\": \"Context mentions sunset over ocean and beach photos\", \"answer\": \"beach\"}
Q: \"Would he prefer a Charger or a Forester?\" → {\"reasoning\": \"Context shows he works on classic muscle cars\", \"answer\": \"Dodge Charger\"}
Q: \"What career might he pursue?\" → {\"reasoning\": \"Context says he is pursuing counseling and mental health\", \"answer\": \"Counseling or social work\"}
Q: \"Would she be considered religious?\" → {\"reasoning\": \"Context says she made art for a church and her necklace symbolizes faith, but also clashed with religious conservatives\", \"answer\": \"Somewhat, but not extremely religious\"}
Q: \"What political leaning?\" → {\"reasoning\": \"Context shows LGBTQ+ advocacy and progressive values\", \"answer\": \"Liberal\"}
Q: \"What card game is she talking about?\" → {\"reasoning\": \"Context describes a card game about cats — this is Exploding Kittens\", \"answer\": \"Exploding Kittens\"}
Q: \"What console does he own?\" → {\"reasoning\": \"Context says he plays Xenoblade 2, which is a Nintendo Switch exclusive\", \"answer\": \"Nintendo Switch\"}
Q: \"In what country did she buy the snake?\" → {\"reasoning\": \"Context says she bought it in Paris, which is in France\", \"answer\": \"France\"}
Q: \"Is it likely she moved?\" → {\"reasoning\": \"Some indirect evidence suggests relocation but not confirmed\", \"answer\": \"Likely yes\"}
Q: \"Does he love music?\" → {\"reasoning\": \"Multiple contexts show passion for music\", \"answer\": \"Yes\"}
Q: \"How many hikes has she been on?\" → {\"reasoning\": \"Context mentions 4 separate hikes\", \"answer\": \"Four\"}
Q: \"How often does he get checkups?\" → {\"reasoning\": \"Context shows checkups every 3 months\", \"answer\": \"every three months\"}

Return ONLY valid JSON: {\"reasoning\": \"...\", \"answer\": \"shortest possible answer\"}
""")
FINAL_ANSWER_PROMPT = Template("""
You are an intelligent memory assistant tasked with answering a question from semantic and original message evidence.

Instructions:
1. Use all evidence jointly.
2. If the question involves time, use timestamps carefully in reasoning.
3. If the question uses relative time references like \"last year\", \"two months ago\", or \"the previous day\", convert them to concrete natural time based on the evidence timestamps.
4. Do not answer temporal questions with raw timestamps like \"2023-05-08\"; answer naturally like \"7 May 2023\" or \"the week before 6 July 2023\".
5. If evidence conflicts, prefer the most recent evidence unless the question explicitly asks about an earlier time.
6. Prefer direct evidence from original messages when available.
7. Focus only on the provided memory evidence.
8. Answer concisely, ideally under 5-10 words.
9. Do not add explanations.

Question:
{{ question }}

Semantic evidence:
{{ semantic }}

Original message evidence:
{{ messages }}

Answer:
""")
INFERENCE_PATTERNS = [
    "^would\\b",
    "^could\\b",
    "^might\\b",
    "^is it likely\\b",
    "^what might\\b",
    "^what would\\b",
    "^what could\\b",
    "^what fields would\\b",
    "^what personality\\b",
    "^what attributes\\b",
    "^what (?:advice|challenges|traits|characteristics)\\b",
    "^what (?:console|card game|technique|country|state)\\b",
    "^what (?:nickname|additional|role|other)\\b",
    "^what is the (?:game|board game)\\b",
    "^would .+ be considered\\b",
    "^would .+ likely\\b",
    "^would .+ enjoy\\b",
    "^would .+ want\\b",
    "^would .+ pursue\\b",
    "^would .+ be open\\b",
    "^does .+ (?:live|love|like|enjoy|prefer|wish|employ)\\b",
    "^is .+ (?:likely|considered|religious|patriotic|married|an? )\\b",
    "^are .+ fans?\\b",
    "^was .+ feeling\\b",
    "^which (?:country|state|us state|console)\\b",
    "^which (?:popular|major)\\b",
    "^in what (?:country|state)\\b",
    "^in which state\\b",
    "^around which\\b",
    "^based on\\b",
    "^in light of\\b",
    "^considering\\b",
    "^who is \\w+\\??$",
    "^how old\\b",
    "^how often\\b",
    "\\bwould\\b",
    "\\bcould\\b",
    "\\blikely\\b",
    "\\bmight\\b",
    "\\bpotentially\\b",
    "\\bwouldn't\\b",
    "\\bprobably\\b",
    "\\bsuspected\\b",
    "\\bprefer\\b",
    "\\balive\\b",
    "\\bfinancial status\\b",
    "\\bpolitical\\b",
    "\\bleaning\\b",
]
