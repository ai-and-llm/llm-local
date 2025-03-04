from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    Pipeline,
)


class LLMModel(ABC):
    @property
    @abstractmethod
    def llm_pipeline(self) -> Pipeline:
        pass


class Phi(LLMModel):
    def __init__(self):
        # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        llm_model_name = "microsoft/Phi-3-mini-4k-instruct"

        print(f"Loading '{llm_model_name}' LLM model")
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        print(f"Loading '{llm_model_name}' tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        print("Creating pipeline")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            return_full_text=False,  # Don't return prompt but only output text
            do_sample=False,
        )

    @property
    def llm_pipeline(self) -> Pipeline:
        return self.pipeline


class Llama(LLMModel):
    def __init__(self):
        # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        print(f"Creating pipeline for '{llm_model_name}'")

        self.pipeline = pipeline(
            "text-generation",
            model=llm_model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=500,
            return_full_text=False,  # Don't return prompt but only output text
        )

    @property
    def llm_pipeline(self) -> Pipeline:
        return self.pipeline


class LLMInstance:
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model

    def ask_question(self, question: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Generate answer to a question in no more than 300 characters or 3 sentences.",
            },
            {"role": "user", "content": f"{question}"},
        ]

        # Generate output
        output = self.llm_model.llm_pipeline(prompt)
        enhanced_sentence = output[0]["generated_text"]

        return enhanced_sentence

    def enhance_sentence(self, sentence: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an advanced AI language model designed to improve the readability of text while preserving its meaning. "
                "Your goal is to rewrite the given input to make it clearer, more engaging, and more natural. "
                "Ensure that the tone is appropriate for the intended audience, sentence structures are fluid, and complex or awkward phrasing is simplified without losing key details. "
                "Generate as an output only enhanced text nothing else.",
            },
            {
                "role": "user",
                "content": f"Here is the text that needs improvement: ${sentence}",
            },
        ]

        # Generate output
        output = self.llm_model.llm_pipeline(messages)
        enhanced_sentence = output[0]["generated_text"]

        return enhanced_sentence

    def classify_movie_review(self, movie_review: str) -> str:
        prompt_template = f"""
        Predict whether the following feedback is a positive or negative movie review:

        ${movie_review}

        If it is positive return 1 and if it is negative return 0. Do not give any other answers.
        """

        prompt = [
            {"role": "user", "content": f"{prompt_template}"},
        ]

        outputs = self.llm_model.llm_pipeline(prompt)

        review_label = outputs[0]["generated_text"].strip()

        return "POSITIVE" if review_label == "1" else "NEGATIVE"
