from chat_completion_wrapper import ChatMessage, LoadingModelError, DisabledEndpointError
import requests
from logger import logger
from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import FinishReason
import dataclasses
from typing import Callable

# resources to keep eyes on
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# https://huggingface.co/blog/inference-endpoints-llm
# https://github.com/philschmid/easyllm -> maybe replace with this library


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class _HFChatCompletionWrapper:
    def __init__(self, endpoint_url: str, token: str, namespace: str, name: str, log: bool = True) -> None:
        self.context_length = None  # should be the same in the container configuration inside the HF endpoint settings
        self.log = log
        self.namespace = namespace
        self.name = name
        self.token = token

        # Streaming Client
        self.client = InferenceClient(endpoint_url, self.token)

        # generation parameter
        self._default_generation_params = None

    def _format_messages(messages: list[ChatMessage]) -> str:
        raise NotImplementedError()

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

        # from https://github.com/philschmid/easyllm/blob/2603d0dc5e3950ea5e645036045f8cc6a46608d0/easyllm/clients/huggingface.py#L134C28-L134C28
        if params["top_p"] == 0:
            params["top_p"] = 2e-4
        if params["top_p"] == 1:
            params["top_p"] = 0.9999999
        if params["temperature"] == 0:
            params["temperature"] = 0.001  # 2e-4

        if "stop" in params:
            # don't use += here, the dict above makes a shallow copy!
            params["stop_sequences"] = params["stop_sequences"] + params["stop"]
            del params["stop"]
            # print("WARNING", params["stop_sequences"])

        # print(self._default_generation_params)
        # print(params)
        return params

    def _check_endpoint_status(self) -> tuple[int, str]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            f"https://api.endpoints.huggingface.cloud/v2/endpoint/{self.namespace}/{self.name}", headers=headers
        )

        return response.status_code, None if response.status_code != 200 else response.json()["status"]["state"]

    def __call__(self, message: str, post_process: Callable[[str], str] | None = None, **kwargs) -> str:
        self.messages.append(ChatMessage(role="user", content=message))
        if self.log:
            logger(content=self.messages[-1])

        params = self._kwargs_to_generation_params(kwargs)
        formatted_messages: str = self._format_messages(self.messages)

        if "max_new_tokens" in params:
            formatted_messages_tokens_length = int(
                len(formatted_messages) // 2.5
            )  # approximation, see: https://github.com/philschmid/easyllm/blob/2603d0dc5e3950ea5e645036045f8cc6a46608d0/easyllm/clients/huggingface.py#L176C9-L176C9
            # print("formatted_messages_tokens_length", formatted_messages_tokens_length)
            # print("params[max_new_tokens]", params["max_new_tokens"])
            if formatted_messages_tokens_length > params["max_new_tokens"]:
                params["max_new_tokens"] = self.context_length - formatted_messages_tokens_length  # max possible value
                # print("updated params[max_new_tokens]", params["max_new_tokens"])
            assert params["max_new_tokens"] > 0

        # Previous attempt
        # https://github.com/huggingface/huggingface_hub/issues/1605#issuecomment-1684105783
        # try:
        #     response = self.client.text_generation(formatted_messages, stream=False, details=True, **params)
        # except HfHubHTTPError as e:
        #     if e.response.status_code == 502:
        #         logger(content="LoadingModelError")
        #         raise LoadingModelError from e
        #     raise
        # except requests.exceptions.ConnectionError as e:
        #     logger(content="UnknownEndpointError")
        #     raise UnknownEndpointError from e

        try:
            response = self.client.text_generation(formatted_messages, stream=False, details=True, **params)
        except Exception as e:
            status_code, endpoint_state = self._check_endpoint_status()
            if status_code == 200:
                # pending, initializing, updating, updateFailed, running, paused, failed, scaledToZero
                if endpoint_state == "paused":
                    raise DisabledEndpointError from e
                elif endpoint_state in ["scaledToZero", "initializing", "updating"]:
                    raise LoadingModelError from e
            raise

        if type(response) is str:
            # TODO temporary to handle gpt4all
            generated_text = response
        else:
            generated_text = response.generated_text
            if response.details.finish_reason == FinishReason.StopSequence:
                generated_text = generated_text.removesuffix(response.details.tokens[-1].text)

        result = generated_text.strip()

        if post_process is not None:
            if self.log:
                logger(observation="assistant-before-post-process", content=ChatMessage(role="assistant", content=result))
            result = post_process(result)

        self.messages.append(ChatMessage(role="assistant", content=result))
        if self.log:
            logger(content=self.messages[-1])

        return result


class HFLlama2ChatCompletionWrapper(_HFChatCompletionWrapper):
    def __init__(self, endpoint_url: str, token: str, namespace: str, name: str, log: bool = True) -> None:
        super().__init__(endpoint_url=endpoint_url, token=token, namespace=namespace, name=name, log=log)
        self.context_length = 4096  # should be the same in the container configuration inside the HF endpoint settings

        # generation parameter
        self._default_generation_params = dict(
            max_new_tokens=1024,
            top_p=1,
            temperature=0,
            stop_sequences=["</s>"],
        )

    def _format_messages(self, messages: list[ChatMessage]) -> str:
        messages = [dataclasses.asdict(msg) for msg in messages]
        if messages[0]["role"] != "system":
            messages = [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                }
            ] + messages
        messages = [
            {
                "role": messages[1]["role"],
                "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
            }
        ] + messages[2:]
        assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
            [msg["role"] == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        formatted_messages: str = "".join(
            [
                f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ]
        )
        assert messages[-1]["role"] == "user", f"Last message must be from user, got {messages[-1]['role']}"
        formatted_messages += f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}"
        return formatted_messages


class HFFalconChatCompletionWrapper(_HFChatCompletionWrapper):
    def __init__(self, endpoint_url: str, token: str, namespace: str, name: str, log: bool = True) -> None:
        super().__init__(endpoint_url=endpoint_url, token=token, namespace=namespace, name=name, log=log)
        self.context_length = 4096  # 2048  # should be the same in the container configuration inside the HF endpoint settings

        # generation parameter
        self._default_generation_params = dict(
            max_new_tokens=1024,
            top_p=1,
            temperature=0,
            stop_sequences=["</s>", "User"],  # I don't think </s> makes sense here, for Falcon
        )

    def _format_messages(self, messages: list[ChatMessage]) -> str:
        if messages[0].role != "system":
            messages.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))

        assert messages[0].role == "system", f"First message must be from system, got {messages[0].role}"
        assert all([msg.role == "user" for msg in messages[1::2]]) and all(
            [msg.role == "assistant" for msg in messages[2::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"

        formatted_messages: str = "\n".join(f"{msg.role.capitalize()}: {msg.content}" for msg in messages)
        formatted_messages += "\nAssistant: "

        return formatted_messages
