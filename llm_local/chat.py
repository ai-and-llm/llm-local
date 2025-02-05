from abc import ABC, abstractmethod

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class LLMModel(ABC):
    @abstractmethod
    def llm_model(self) -> PreTrainedModel:
        pass

    @abstractmethod
    def llm_tokenizer(self) -> PreTrainedTokenizer:
        pass


class Phi(LLMModel):
    def __init__(self):
        # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        llm_model_name = "microsoft/Phi-3-mini-4k-instruct"

        print(f"Loading '{llm_model_name}' LLM model")
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)

        print(f"Loading '{llm_model_name}' tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    def llm_model(self) -> PreTrainedModel:
        return self.model

    def llm_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer


class Llama(LLMModel):
    def __init__(self):
        # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        print(f"Loading '{llm_model_name}' LLM model")
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)

        print(f"Loading '{llm_model_name}' tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    def llm_model(self) -> PreTrainedModel:
        return self.model

    def llm_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer


class Chat:
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model

    def ask_question(self, question: str) -> str:
        # Create a pipeline
        generator = pipeline(
            "text-generation",
            model=self.llm_model.llm_model(),
            tokenizer=self.llm_model.llm_tokenizer(),
            max_new_tokens=500,
            return_full_text=False,  # Don't return prompt but only output text
        )

        prompt = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"{question}"},
        ]

        # Generate output
        output = generator(prompt)
        enhanced_sentence = output[0]["generated_text"]

        return enhanced_sentence
