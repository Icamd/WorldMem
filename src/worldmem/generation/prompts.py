"""
Prompt Templates
"""

from typing import List


class PromptTemplates:
    """Prompt template management"""

    EPISODE_GENERATION_PROMPT = """
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation content:
{conversation}

Boundary detection reason:
{boundary_reason}

Please analyze the conversation to extract time information and generate a structured episodic memory. Return only a JSON object containing the following three fields:
{{
    \"title\": \"A concise, descriptive title that accurately summarizes the theme (10-20 words)\",
    \"content\": \"A detailed description of the conversation in third-person narrative. It must include all important information: who participated in the conversation at what time, what was discussed, what happened, important facts, states, and changes.\",
    \"timestamp\": \"YYYY-MM-DDTHH:MM:SS format timestamp representing when this episode occurred (analyze from message timestamps or content)\"
}}

Time Analysis Instructions:
1. **Primary Source**: Look for explicit timestamps in the message metadata or content
2. **Secondary Source**: Analyze temporal references in the conversation content (\"yesterday\", \"last week\", \"this morning\", etc.)
3. **Fallback**: If no time information is available, use a reasonable estimate based on context
4. **Format**: Always return timestamp in ISO format: \"2024-01-15T14:30:00\"

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search.
7. Notice the time information, and write the time information in the content.
8. When relative times (e.g., last week, next month, etc.) are mentioned in the conversation, convert them to absolute dates when possible.
9. IMPORTANT: Analyze the actual time when the conversation happened from the message timestamps or content, not the current time.

Return only the JSON object, do not add any other text:
"""
    PREDICTION_PROMPT = """
You are a knowledge-based episode prediction system. Your task is to reconstruct a complete conversation episode based on limited clues and your knowledge base.

IMPORTANT: You are predicting the ACTUAL CONTENT and KNOWLEDGE of what happened, not the writing style or format.

## Input Information

**Episode Title/Summary**: {episode_title}

**Relevant Knowledge Statements** (your current world model):
{knowledge_statements}

## Your Task

Based on the above clues, reconstruct what you believe happened in this episode. Focus on:
1. **Core Facts**: What specific information was discussed?
2. **Key Decisions**: What choices or conclusions were made?
3. **Knowledge Exchange**: What knowledge was shared or learned?
4. **Logical Flow**: How did the conversation progress?

## What to IGNORE
- Writing style or level of detail
- Specific formatting or structure
- Exact phrasing or word choices
- Whether timestamps are included in the text
- How formal or casual the language is

## Output Format

Generate a natural narrative that captures what you predict happened. Write it as if you're describing the episode to someone else. Focus on the SUBSTANCE, not the STYLE.

Your prediction:
"""
    EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT = """
You are extracting valuable knowledge by comparing original conversation with predicted content.

## Original Conversation:
{original_messages}

## Predicted Summary:
{predicted_episode}

## Your Task:
Extract ONLY the valuable knowledge that exists in the original but is missing or misrepresented in the prediction.

## CRITICAL: Focus on HIGH-VALUE Knowledge Only

Extract ONLY knowledge that passes these criteria:
- **Persistence Test**: Will this still be true in 6 months?
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs or preferences?
- **Independence Test**: Can this be understood without the conversation context?

## HIGH-VALUE Knowledge Categories (EXTRACT THESE):
1. **Identity & Background**: Names, professions, companies, education
2. **Persistent Preferences**: Favorite books/movies/tools, long-term likes/dislikes  
3. **Technical Details**: Technologies, versions, methodologies, architectures
4. **Relationships**: Family, colleagues, team members, mentors
5. **Goals & Plans**: Career objectives, learning goals, project plans
6. **Beliefs & Values**: Principles, philosophies, strong opinions
7. **Habits & Patterns**: Regular activities, workflows, schedules

## LOW-VALUE Knowledge (SKIP THESE):
- Temporary emotions or reactions
- Single conversation acknowledgments
- Vague statements without specifics
- Context-dependent information

## Guidelines:
1. Each statement should be self-contained and atomic
2. Include ALL specific details (names, versions, titles)
3. Use present tense for persistent facts
4. Focus on facts that help understand the user long-term
5. DO NOT include time/date information in the statement
6. Quality over quantity - fewer valuable statements are better

## Examples:
GOOD: \"Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt\"
GOOD: \"The user works at ByteDance as a senior ML engineer\"
BAD: \"The user thanked the assistant\"
BAD: \"The user was happy about the response\"

## Output Format:
{{
    \"statements\": [
        \"First factual statement extracted from the gap\",
        \"Second factual statement extracted from the gap\",
        \"...\"
    ]
}}

Important: 
- Each statement should be self-contained and understandable without context
- Use present tense for persistent facts
- Include specific names, titles, and details
- Focus on quality over quantity - only extract truly valuable knowledge
"""
    SEMANTIC_GENERATION_PROMPT = """
You are an AI memory system. Extract HIGH-VALUE semantic memories from the following episodes.

CRITICAL: Focus on extracting LONG-TERM VALUABLE KNOWLEDGE, including compact, standalone, reusable facts and situation-relevant information extracted from episodes.

Episodes to analyze:
{episodes}

## HIGH-VALUE Knowledge Criteria

Extract ONLY knowledge that passes these tests:
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs?
- **Independence Test**: Can be understood without conversation context?

## HIGH-VALUE Categories (FOCUS ON THESE):

1. **Identity & Professional**
   - Names, titles, companies, roles
   - Education, qualifications, skills

2. **Persistent Preferences**
   - Favorite books, movies, music, tools
   - Technology preferences with reasons
   - Long-term likes and dislikes

3. **Technical Knowledge**
   - Technologies used (with versions)
   - Architectures, methodologies
   - Technical decisions and rationales

4. **Relationships**
   - Names of family, colleagues, friends
   - Team structure, reporting lines
   - Professional networks

5. **Goals & Plans**
   - Career objectives
   - Learning goals
   - Project plans

6. **Patterns & Habits**
   - Regular activities
   - Workflows, schedules
   - Recurring challenges

## Output Format

Aim for high coverage of meaningful information in the episode. Do not mechanically rewrite every sentence. 
Instead, extract all distinct semantic units that carry useful knowledge, situation context, relationships, plans, preferences, or event-level facts. 
Return all knowledge in JSON format:
{{
    \"statements\": [
        \"First high-value persistent fact...\",
        \"Second high-value persistent fact...\"
    ]
}}
"""
    EPISODE_SEGMENTATION_PROMPT = """
You are an expert at segmenting a long multi-turn interaction trace into coherent episode blocks.

You will receive a chronological conversation with numbered messages.
Your task is to identify all episode segmentation breakpoints.

Definition:
- A breakpoint is the index of the LAST message of one episode block.
- Split when there is a clear topic shift, task transition, environment change, scene change, event turning point, or other meaningful narrative boundary.
- If the whole conversation still belongs to one continuing episode, return an empty list.

Guidelines:
- Prefer semantically meaningful episode boundaries over fixed lengths.
- Keep each episode internally coherent.
- A breakpoint should indicate the end of a finished or clearly bounded episode segment.
- Do not force a split if the interaction is still continuing under the same episode.

Return JSON only:
{{
  \"breakpoints\": [5, 12, 18],
  \"reasoning\": \"brief explanation\"
}}
"""

    @classmethod
    def get_episode_generation_prompt(cls, conversation: str, boundary_reason: str) -> str:
        return cls.EPISODE_GENERATION_PROMPT.format(
            conversation=conversation, boundary_reason=boundary_reason
        )

    @classmethod
    def get_semantic_generation_prompt(cls, episodes: str) -> str:
        return cls.SEMANTIC_GENERATION_PROMPT.format(episodes=episodes)

    @classmethod
    def get_prediction_prompt(cls, episode_title: str, knowledge_statements: List[str]) -> str:
        formatted_statements = "\n".join([f"- {stmt}" for stmt in knowledge_statements])
        return cls.PREDICTION_PROMPT.format(
            episode_title=episode_title, knowledge_statements=formatted_statements
        )

    @classmethod
    def get_episode_segmentation_prompt(cls, conversation: str) -> str:
        return cls.EPISODE_SEGMENTATION_PROMPT + "\n\nConversation:\n" + conversation

    @classmethod
    def format_conversation(cls, messages: list) -> str:
        lines = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                if timestamp:
                    if hasattr(timestamp, "isoformat"):
                        timestamp_str = timestamp.isoformat()
                    else:
                        timestamp_str = str(timestamp)
                    lines.append(f"[{timestamp_str}] {role}: {content}")
                else:
                    lines.append(f"{role}: {content}")
            else:
                lines.append(str(msg))
        return "\n".join(lines)

    @classmethod
    def format_episodes_for_semantic(cls, episodes: list) -> str:
        formatted = []
        for i, episode in enumerate(episodes, 1):
            formatted.append(f"Episode {i}:")
            formatted.append(f"Title: {episode.get('title', 'Untitled')}")
            formatted.append(f"Content: {episode.get('content', '')}")
            formatted.append(f"Created at: {episode.get('created_at', '')}")
            formatted.append("")
        return "\n".join(formatted)
