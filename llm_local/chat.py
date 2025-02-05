from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Chat:
    def __init__(self):
        # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        llm_model_name = "microsoft/Phi-3-mini-4k-instruct"

        print(f"Loading '{llm_model_name}' LLM model")
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)

        print(f"Loading '{llm_model_name}' tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    def ask_question(self, question: str) -> str:
        # Create a pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            return_full_text=False,  # Don't return prompt but only output text
            do_sample=False,
        )

        prompt = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"{question}"},
        ]

        # Generate output
        output = generator(prompt)
        enhanced_sentence = output[0]["generated_text"]

        return enhanced_sentence
