"""Tests for the Agent (with mocked Anthropic client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from second_brain.core.agent import Agent, Conversation
from second_brain.ltm.retriever import RetrievalResult, Retriever


class TestConversation:
    def test_add_and_format(self):
        conv = Conversation()
        conv.add("user", "hello")
        conv.add("assistant", "hi there")
        msgs = conv.to_api_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def test_empty_conversation(self):
        conv = Conversation()
        assert conv.to_api_messages() == []


class TestAgent:
    @patch("second_brain.core.agent.anthropic.Anthropic")
    def test_chat_with_context(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_client.messages.create.return_value = mock_response

        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.search.return_value = RetrievalResult(
            memories=[], context_text="some context"
        )

        settings = MagicMock()
        settings.anthropic_api_key = "test"
        settings.llm.model = "claude-sonnet-4-20250514"
        settings.llm.max_tokens = 100

        agent = Agent(retriever=mock_retriever, settings=settings)
        result = agent.chat("test question")

        assert result == "Test response"
        mock_retriever.search.assert_called_once_with("test question")
        mock_client.messages.create.assert_called_once()

    @patch("second_brain.core.agent.anthropic.Anthropic")
    def test_chat_without_context(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="No context response")]
        mock_client.messages.create.return_value = mock_response

        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.search.return_value = RetrievalResult(
            memories=[], context_text=""
        )

        settings = MagicMock()
        settings.anthropic_api_key = "test"
        settings.llm.model = "claude-sonnet-4-20250514"
        settings.llm.max_tokens = 100

        agent = Agent(retriever=mock_retriever, settings=settings)
        result = agent.chat("test question")

        assert result == "No context response"
        # Verify the message doesn't contain <context> tags when no context
        call_kwargs = mock_client.messages.create.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "<context>" not in user_msg

    @patch("second_brain.core.agent.anthropic.Anthropic")
    def test_reset_clears_history(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_client.messages.create.return_value = mock_response

        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.search.return_value = RetrievalResult(memories=[], context_text="")

        settings = MagicMock()
        settings.anthropic_api_key = "test"
        settings.llm.model = "test-model"
        settings.llm.max_tokens = 100

        agent = Agent(retriever=mock_retriever, settings=settings)
        agent.chat("first message")
        assert len(agent._conversation.messages) == 2  # user + assistant

        agent.reset()
        assert len(agent._conversation.messages) == 0
