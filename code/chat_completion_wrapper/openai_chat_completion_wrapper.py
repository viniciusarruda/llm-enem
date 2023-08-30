import os
import openai
from typing import Callable
import random
import time
from chat_completion_wrapper import ChatMessage, AuthenticationError
from logger import logger
import dataclasses


class OpenAIChatCompletionWrapper:
    def __init__(self, model: str, log: bool = True, openai_api_key: str | None = None) -> None:
        if openai_api_key is None:
            assert "OPENAI_API_KEY" in os.environ
        else:
            # assert "OPENAI_API_KEY" not in os.environ
            openai.api_key = openai_api_key
        self.check_api_key()

        self.log = log

        # generation parameter
        self._default_generation_params = dict(
            model=model,
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        self.openai_errors: tuple = (
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
        )

    def check_api_key(self):
        try:
            openai.Model.list()
        except openai.error.AuthenticationError as e:
            raise AuthenticationError from e

    def new_session(self, system_content: str | None = None, messages: list[ChatMessage] | None = None):
        self.messages: list[ChatMessage] = []

        if self.log:
            logger(content=" --- NEW CHAT COMPLETION SESSION --- ")

        if system_content is not None:
            assert messages is None
            self.messages.append(ChatMessage(role="system", content=system_content))
            if self.log:
                logger(content=self.messages[-1])

        elif messages is not None:
            self.messages = messages
            if self.log:
                for msg in self.messages:
                    logger(content=msg)

    def _kwargs_to_generation_params(self, kwargs: dict) -> dict:
        params = dict(self._default_generation_params, **kwargs)  # overwriting default parameters
        return params

    def _completions_with_backoff(
        self,
        params: dict = {},
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
    ) -> dict:
        # from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb

        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return openai.ChatCompletion.create(
                    messages=[dataclasses.asdict(msg) for msg in self.messages],
                    **params,
                    request_timeout=60,
                )

            # Retry on specified errors
            except self.openai_errors as e:
                if self.log:
                    logger(observation="Error", content=str(e))
                    logger.save()

                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                if self.log:
                    logger(observation="Info", content=f"Waiting {delay} before another attempt.")
                    logger.save()
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    def __call__(self, message: str, post_process: Callable[[str], str] | None = None, **kwargs) -> str:
        self.messages.append(ChatMessage(role="user", content=message))
        if self.log:
            logger(content=self.messages[-1])

        params = self._kwargs_to_generation_params(kwargs)
        completion = self._completions_with_backoff(params=params)
        result = completion.choices[0].message.content
        result = result.strip()

        if post_process is not None:
            if self.log:
                logger(
                    observation="assistant-before-post-process",
                    content=ChatMessage(role="assistant", content=result),
                )
            result = post_process(result)

        self.messages.append(ChatMessage(role="assistant", content=result))
        if self.log:
            logger(content=self.messages[-1])

        return result
