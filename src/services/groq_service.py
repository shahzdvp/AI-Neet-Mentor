"""
src/services/groq_service.py — Groq LLM Service
=================================================
WHY Groq?
- Groq runs LLMs on custom hardware (LPUs) → extremely fast inference.
- Free tier: ~14,400 requests/day with llama-3.1-70b.
- Simple API: identical structure to OpenAI's API (easy to swap later).

WHAT IS THE LLM'S ROLE IN RAG?
  The retrieved NCERT chunks are NOT the final answer.
  They are the CONTEXT that the LLM uses to generate the answer.

  Think of it like an open-book exam:
  - The student (LLM) gets to look at specific NCERT pages (retrieved chunks)
  - Then writes an answer in their own words
  - The answer is grounded in NCERT, not hallucinated

MESSAGE FORMAT (OpenAI-style):
  [
    {"role": "system",    "content": "You are a NEET tutor..."},
    {"role": "user",      "content": "Previous question..."},     ← history
    {"role": "assistant", "content": "Previous answer..."},       ← history
    {"role": "user",      "content": "NCERT context:\n...\n\nQuestion: ..."}  ← current
  ]

ERROR HANDLING:
  LLM APIs can fail (rate limits, timeouts, API errors).
  We catch exceptions and return a graceful error message instead
  of crashing the server.
"""

import logging
from groq import Groq, APIError, RateLimitError

logger = logging.getLogger(__name__)

# Fallback message when the LLM call fails.
_FALLBACK_RESPONSE = (
    "I'm having trouble connecting right now. "
    "Please try again in a moment."
)


class GroqService:
    """
    Wrapper around the Groq Python client.
    Isolates all LLM communication in one place.
    """

    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        if not api_key:
            logger.warning(
                "GROQ_API_KEY is not set. LLM calls will fail. "
                "Add it to your .env file."
            )
        # The Groq() client reads the key and stores it internally.
        self._client      = Groq(api_key=api_key)
        self._model       = model
        self._max_tokens  = max_tokens
        self._temperature = temperature

    def complete(self, messages: list[dict]) -> str:
        """
        Sends a list of messages to Groq and returns the assistant's reply.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
                      Built by PromptBuilder.

        Returns:
            The LLM's reply as a plain string.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                # stream=False → wait for the full response.
                # For streaming (word-by-word like ChatGPT), set stream=True
                # and yield tokens — a more advanced feature for later.
            )

            # The response object has nested structure.
            # .choices[0].message.content is the actual text reply.
            answer = response.choices[0].message.content
            logger.debug("LLM responded (%d chars)", len(answer))
            return answer

        except RateLimitError:
            logger.error("Groq rate limit hit. Slow down requests.")
            return "I've hit my rate limit. Please wait a minute and try again."

        except APIError as e:
            logger.error("Groq API error: %s", e)
            return _FALLBACK_RESPONSE

        except Exception as e:
            logger.error("Unexpected LLM error: %s", e)
            return _FALLBACK_RESPONSE
