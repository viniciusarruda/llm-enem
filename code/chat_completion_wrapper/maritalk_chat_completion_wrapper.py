import maritalk
from typing import Callable
from chat_completion_wrapper import ChatMessage
from logger import logger
import dataclasses


class MariTalkChatCompletionWrapper:
    def __init__(self, log: bool = True) -> None:
        self.model = maritalk.MariTalk(
            key="110465859132359652937$80e7f77f3f51e284d2da454306b16ebb589dae350b6e9abe1bbaf0ec66c10356"
        )
        self.log = log

        # generation parameter
        self._default_generation_params = dict(
            temperature=0,
            max_tokens=1024,
            top_p=0.95,
            do_sample=True,
            stopping_tokens=[],
        )

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

        if "stop" in params:
            # don't use += here, the dict above makes a shallow copy!
            params["stopping_tokens"] = params["stopping_tokens"] + params["stop"]
            del params["stop"]

        return params

    def _chat_completion(self, params: dict = {}) -> dict:
        """function to call the endpoint, handling possible issues and trying again in case of failure
        It doesn't do any processing
        """
        return self.model.generate(
            messages=[dataclasses.asdict(msg) for msg in self.messages],
            **params,
        )

    def __call__(self, message: str, post_process: Callable[[str], str] | None = None, **kwargs) -> str:
        self.messages.append(ChatMessage(role="user", content=message))
        if self.log:
            logger(content=self.messages[-1])

        params = self._kwargs_to_generation_params(kwargs)
        completion = self._chat_completion(params=params)
        result = completion.strip()

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
