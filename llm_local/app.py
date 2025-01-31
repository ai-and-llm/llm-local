from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main() -> None:

    llm_model_name = "microsoft/Phi-3-mini-4k-instruct"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        # device_map="cuda",
        # torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    # Create a pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, # Don't return prompt but only output text
        max_new_tokens=500,
        do_sample=False
    )

    # The prompt (user input / query)
    messages = [
        {"role": "user", "content": "Rust vs Java comparison for writing backend systems."}
    ]

    # Generate output
    output = generator(messages)
    generated_response = output[0]["generated_text"]
    print(generated_response)


if __name__ == '__main__':
    main()
