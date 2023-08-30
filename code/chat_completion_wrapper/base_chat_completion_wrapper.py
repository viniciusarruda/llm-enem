from dataclasses import dataclass
from typing import Literal


class LoadingModelError(Exception):
    pass


class DisabledEndpointError(Exception):
    pass


class AuthenticationError(Exception):
    pass


@dataclass  # slots?
class ChatMessage:
    role: Literal["system", "assistant", "user"]
    content: str  # | list | bool

    def __post_init__(self):
        if self.role not in ["system", "assistant", "user"]:
            raise TypeError('role should be "system", "assistant", or "user"')

        if not isinstance(self.content, str):
            raise TypeError("content should be of type str")

        # if not isinstance(self.age, int):
        #     raise TypeError("Age should be of type int")

        # if self.age < 0 or self.age > 150:
        #     raise ValueError("Age must be between 0 and 150")


# if __name__ == "__main__":
#     ChatMessage(role=Role.User, content="asdasd")
#     assert "user" == Role("user") == Role.User == Role(Role.User)
