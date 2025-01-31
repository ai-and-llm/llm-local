import transformers


def chat(question: str) -> None:
    #
    # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
    #
    llm_model_name = "microsoft/Phi-3-mini-4k-instruct"

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_model_name)

    # Create a pipeline
    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,  # Don't return prompt but only output text
        max_new_tokens=500,
        do_sample=False,
    )

    # The prompt (user input / query)
    # messages = [
    #     {"role": "user", "content": f"Rewrite the following text to make it more human-readable, clear, and concise. Avoid jargon and complex sentences. Focus on making the message easy to understand for a general audience. Text: {question}"},
    # ]

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

    # Generate output
    output = generator(messages)
    print(f"Phi response: ========> {output[0]['generated_text']}")
