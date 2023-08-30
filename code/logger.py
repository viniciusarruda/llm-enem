import os
import json
import uuid
from jinja2 import Environment, FileSystemLoader
from chat_completion_wrapper import ChatMessage
from dataclasses import dataclass, field
import html
from datetime import datetime

from pathlib import Path

# TODO
# validation (if suitable and necessary)
# https://github.com/pytorch/pytorch/blob/7fb543e36ddca3b96543bce857f76fbeef64fda5/torch/distributed/_shard/sharded_tensor/shard.py#L8
# https://stackoverflow.com/questions/50563546/validating-detailed-types-in-python-dataclasses


RESET_CONSOLE_COLOR = "\033[00m"

CONSOLE_COLOR_MAP = {
    "reset": "\033[00m",
    "system": ("\033[1;35m", "\033[35m"),
    "user": ("\033[1;33m", "\033[33m"),
    "assistant": ("\033[1;31m", "\033[31m"),
}

HTML_COLOR_MAP = {
    "system": ("indigo", "indigo"),
    "user": ("darkorange", "darkorange"),
    "assistant": ("darkred", "darkred"),
    "assistant-before-post-process": ("brown", "brown"),
}


# next step dont need to create a class log message, just log it directly!


@dataclass
class LogMessage:
    # TODO validation
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    observation: str | None = None

    def _stringfy(self, obj: str | bool | list | dict) -> str:
        if type(obj) in [list, dict]:
            obj = json.dumps(obj, sort_keys=True, indent=4)
        elif type(obj) is bool:
            obj = str(obj)

        if type(obj) is not str:
            print("WARNING: OBJ NOT STRING!!!!")
            print(obj)
            obj = str(obj)
        return obj

    def to_console(self) -> str:
        other_color = "\033[1;34m"
        content = self._stringfy(self.content)
        return f"{other_color}{content}{RESET_CONSOLE_COLOR}"

    def to_html(self) -> str:
        content = self._stringfy(self.content)
        content = html.escape(content)
        # return f'<pre style="color:blue">{content}</pre>'
        return content


@dataclass  # slots?
class LogChatMessage:
    # TODO validation
    chat_message: ChatMessage
    timestamp: datetime = field(default_factory=datetime.now)
    observation: str | None = None

    def to_console(self) -> str:
        role_color, content_color = CONSOLE_COLOR_MAP[self.chat_message.role]
        return f"{role_color}{self.chat_message.role.upper()}{RESET_CONSOLE_COLOR}> {content_color}{self.chat_message.content}{RESET_CONSOLE_COLOR}"

    def to_html(self) -> str:
        role_color, content_color = HTML_COLOR_MAP[self.chat_message.role]
        content = html.escape(self.chat_message.content)
        # return f'<span style="color:{role_color};font-weight:bold;">{self.chat_message.role.upper()}</span>> <pre style="color:{content_color}">{content}</pre>'
        return f"<b>{self.chat_message.role.upper()}</b>> {content}"


class _Logger:
    _counter = 0

    def __init__(self, print_stdout: bool = False) -> None:
        _Logger._counter += 1
        assert _Logger._counter == 1, "Logger should not be instantiated out of this file!"
        env = Environment(loader=FileSystemLoader(os.path.join(Path(__file__).parent.parent.resolve(), "log-templates")))
        self.template = env.get_template("template.jinja2")
        self.print_stdout = print_stdout
        self.log_messages: list[LogMessage | LogChatMessage] = []

    def __call__(self, content: str | ChatMessage, observation: str | None = None) -> None:
        if type(content) is ChatMessage:
            log_message = LogChatMessage(chat_message=content, observation=observation)
        else:
            log_message = LogMessage(content=content, observation=observation)

        if self.print_stdout:
            print(log_message.to_console())
        # self.logger.info(log_message.to_html())
        self.log_messages.append(log_message)

    def save(self) -> str:
        log_filepath = os.path.join(Path(__file__).parent.parent.resolve(), "log", f"{str(uuid.uuid4())}.html")
        output = self.template.render(log_messages=self.log_messages)
        # Write the rendered output to an HTML file
        with open(log_filepath, "w", encoding="utf-8") as file:
            file.write(output)
        self.log_messages = []
        return log_filepath


# singleton like! use this, do not instanciate another Logger
logger = _Logger(print_stdout=False)


# colocar tipo do dado! do content ?
