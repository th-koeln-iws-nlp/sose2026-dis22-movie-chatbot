from llm.llm_interface import LLMInterface


class DummyLLM(LLMInterface):
    def generate_response(self, messages: list) -> str:
        return "I'm a dummy LLM. I always say the same thing. Ask me anything about movies!"
