from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, messages: list) -> str:
        pass
