from llm.llm_interface import LLMInterface


class DummyLLM(LLMInterface):
    def generate_response(self, prompt: str) -> str:
        return "I'm a dummy LLM. I always say the same thing. Ask me anything about movies!"
