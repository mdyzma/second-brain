"""Core agent: RAG chat loop with LTM retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from second_brain.config.settings import AppSettings
from second_brain.ltm.retriever import Retriever

SYSTEM_PROMPT = """\
You are a personal knowledge assistant ("Second Brain").
You have access to the user's long-term memory containing their notes,
ideas, and documents. Use the retrieved context to answer questions
accurately. If the context does not contain relevant information,
say so honestly.

When referencing information from context, cite the source number [1], [2], etc."""


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def to_api_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


class Agent:
    """Phase 1 agent: retrieve context from LTM, call Claude, return response."""

    def __init__(
        self,
        retriever: Retriever,
        settings: AppSettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._settings = settings or AppSettings()
        self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        self._conversation = Conversation()

    def chat(self, user_input: str) -> str:
        """Process a single user turn. Returns assistant response text."""
        # 1. Retrieve relevant context from LTM
        retrieval = self._retriever.search(user_input)

        # 2. Build the user message with context
        if retrieval.context_text:
            augmented_input = (
                f"<context>\n{retrieval.context_text}\n</context>\n\n"
                f"User question: {user_input}"
            )
        else:
            augmented_input = user_input

        # 3. Add to conversation history
        self._conversation.add("user", augmented_input)

        # 4. Call Claude
        response = self._client.messages.create(
            model=self._settings.llm.model,
            max_tokens=self._settings.llm.max_tokens,
            system=SYSTEM_PROMPT,
            messages=self._conversation.to_api_messages(),
        )
        assistant_text = response.content[0].text

        # 5. Store response in conversation
        self._conversation.add("assistant", assistant_text)

        return assistant_text

    def reset(self) -> None:
        """Clear conversation history."""
        self._conversation = Conversation()
