"""
Episode Generator
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from ..clients import LLMClient
from ..config import MemoryConfig
from ..models import Episode, Message
from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class EpisodeGenerator:
    """Episode generator for creating episodic memories from conversations"""

    def __init__(self, llm_client: LLMClient, config: MemoryConfig):
        """
        Initialize episode generator

        Args:
            llm_client: LLM client for generation
            config: Memory system configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.prompts = PromptTemplates()
        logger.info("Episode generator initialized")

    def generate_episode(
        self, user_id: str, messages: List[Message], boundary_reason: str = ""
    ) -> Episode:
        """
        Generate an episode from a list of messages

        Args:
            user_id: User ID
            messages: List of messages to convert to episode
            boundary_reason: Reason for episode boundary

        Returns:
            Generated episode
        """
        try:
            if not messages:
                raise ValueError("Cannot generate episode from empty message list")
            conversation = self.prompts.format_conversation(
                [msg.to_dict(clean_metadata=False) for msg in messages]
            )
            episode_data = self._generate_episode_with_llm(conversation, boundary_reason)
            episode_timestamp = self._determine_episode_timestamp(episode_data, messages)
            episode = Episode(
                user_id=user_id,
                title=episode_data.get("title", "Untitled Episode"),
                content=episode_data.get("content", "No content generated"),
                original_messages=[msg.to_dict(clean_metadata=True) for msg in messages],
                message_count=len(messages),
                boundary_reason=boundary_reason,
                timestamp=episode_timestamp,
            )
            logger.info(
                f"Generated episode: {episode.title} ({len(messages)} messages) at {episode_timestamp}"
            )
            return episode
        except Exception as e:
            logger.error(f"Error generating episode: {e}")
            raise

    def _generate_episode_with_llm(self, conversation: str, boundary_reason: str) -> Dict[str, Any]:
        """
        Generate episode using LLM

        Args:
            conversation: Formatted conversation text
            boundary_reason: Reason for episode boundary

        Returns:
            Episode data dictionary
        """
        try:
            prompt = self.prompts.get_episode_generation_prompt(
                conversation=conversation, boundary_reason=boundary_reason
            )
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024 * 8,
                default_response={
                    "title": "Episode generation failed",
                    "content": "Unable to generate episode content due to an error.",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            if not isinstance(response, dict):
                raise ValueError("Invalid response format")
            required_fields = ["title", "content", "timestamp"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                logger.warning(f"Missing fields in LLM response: {missing_fields}")
                if "title" not in response or "content" not in response:
                    raise ValueError("Missing required fields in response")
            title = str(response["title"]).strip()
            if not title:
                title = "Generated Episode"
            content = str(response["content"]).strip()
            if not content:
                content = "Episode content not generated"
            timestamp = None
            if "timestamp" in response:
                timestamp_str = str(response["timestamp"]).strip()
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        logger.debug(f"Parsed timestamp from LLM: {timestamp}")
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid timestamp format from LLM: {timestamp_str}, error: {e}"
                        )
                        timestamp = None
            logger.debug(f"Generated episode: {title}")
            result = {"title": title, "content": content}
            if timestamp:
                result["timestamp"] = timestamp
            return result
        except Exception as e:
            logger.error(f"Error in LLM episode generation: {e}")
            raise

    def _determine_episode_timestamp(
        self, episode_data: Dict[str, Any], messages: List[Message]
    ) -> datetime:
        """
        Determine the timestamp for the episode

        Args:
            episode_data: Episode data from LLM
            messages: Original messages

        Returns:
            Timestamp for the episode
        """
        if "timestamp" in episode_data and episode_data["timestamp"]:
            llm_timestamp = episode_data["timestamp"]
            if isinstance(llm_timestamp, datetime):
                logger.debug(f"Using LLM-generated timestamp: {llm_timestamp}")
                return llm_timestamp
        if messages:
            try:
                earliest_message = min(messages, key=lambda msg: msg.timestamp)
                logger.debug(f"Using earliest message timestamp: {earliest_message.timestamp}")
                return earliest_message.timestamp
            except (AttributeError, ValueError) as e:
                logger.warning(f"Error getting message timestamps: {e}")
        current_time = datetime.now()
        logger.warning(f"Using current time as fallback timestamp: {current_time}")
        return current_time
