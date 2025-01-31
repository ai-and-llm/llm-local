import torch
import transformers


def chat(question: str) -> None:
    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    generator = transformers.pipeline(
        "text-generation",
        model=llm_model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": "You are an advanced AI language model designed to improve the readability of text while preserving its meaning. Your goal is to rewrite the given input to make it clearer, more engaging, and more natural. Ensure that the tone is appropriate for the intended audience, sentence structures are fluid, and complex or awkward phrasing is simplified without losing key details",
        },
        {
            "role": "user",
            "content": f"Here is the text that needs improvement: ${question}",
        },
    ]

    # messages = [
    #     {"role": "user", "content": f"Rewrite the following text to make it more human-readable, clear, and concise. Avoid jargon and complex sentences. Focus on making the message easy to understand for a general audience. Text: {question}"},
    # ]

    outputs = generator(
        messages,
        max_new_tokens=256,
    )
    print(f"LLama response: ========> {outputs[0]['generated_text'][-1]['content']}")


def classification(document: str) -> str:
    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    generator = transformers.pipeline(
        "text-generation",
        model=llm_model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        return_full_text=False,  # Don't return prompt but only output text
    )

    prompt_template = f"""
    Predict whether the following document is a positive or negative movie review:

    ${document}

    If it is positive return 1 and if it is negative return 0. Do not give any other answers.
    """

    prompt = [
        {"role": "user", "content": f"{prompt_template}"},
    ]

    outputs = generator(
        prompt,
        max_new_tokens=256,
    )

    review_label = outputs[0]["generated_text"]

    return "POSITIVE" if review_label == "1" else "NEGATIVE"
