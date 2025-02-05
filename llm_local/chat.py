# Define an abstract base class

import transformers


class Chat:

    def __init__(self):
        # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        llm_model_name = "microsoft/Phi-3-mini-4k-instruct"

        print(f"Loading '{llm_model_name}' LLM model")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        print(f"Loading '{llm_model_name}' tokenizer")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_model_name)

    def ask_question(self, question: str) -> str:

        # Create a pipeline
        generator = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,  # Don't return prompt but only output text
            max_new_tokens=500,
            do_sample=False,
        )

        prompt = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", f"content": f"{question}"},
        ]

        # Generate output
        output = generator(prompt)
        enhanced_sentence = output[0]["generated_text"]

        return enhanced_sentence

