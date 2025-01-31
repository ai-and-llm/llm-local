import os

import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import login

def load_envs() -> None:
    load_dotenv()
    hugging_face_token = os.getenv('HUGGINGFACE_TOKEN')
    login(hugging_face_token)

def main() -> None:
    load_envs()

    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # model = AutoModelForCausalLM.from_pretrained(
    #     llm_model_name,
    #     device_map="auto",
    #     torch_dtype="auto",
    #     trust_remote_code=True,
    model_kwargs = {"torch_dtype": torch.bfloat16},
    # )

    generator = transformers.pipeline(
        "text-generation",
        model=llm_model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    text = "the Merge  button is not available for merge_requests/40 even taking into account that MR was properly approved. Can you merge mentioned MR?"

    messages = [
        {"role": "system", "content": "You are an advanced AI language model designed to improve the readability of text while preserving its meaning. Your goal is to rewrite the given input to make it clearer, more engaging, and more natural. Ensure that the tone is appropriate for the intended audience, sentence structures are fluid, and complex or awkward phrasing is simplified without losing key details"},
        {"role": "user", "content": f"Here is the text that needs improvement: ${text}"},
    ]

    outputs = generator(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])

    # llm_model_name = "microsoft/Phi-3-mini-4k-instruct"
    #
    # # Load model and tokenizer
    # model = AutoModelForCausalLM.from_pretrained(
    #     llm_model_name,
    #     device_map="auto",
    #     torch_dtype="auto",
    #     trust_remote_code=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    #
    # # Create a pipeline
    # generator = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     return_full_text=False, # Don't return prompt but only output text
    #     max_new_tokens=500,
    #     do_sample=False
    # )
    #
    # text = "the Merge  button is not available for merge_requests/40 even taking into account that MR was properly approved. Can you merge mentioned MR?";
    #
    # # The prompt (user input / query)
    # messages = [
    #     {"role": "user", "content": f"Rewrite the following text to make it more human-readable, clear, and concise. Avoid jargon and complex sentences. Focus on making the message easy to understand for a general audience. Text: {text}"},
    # ]
    #
    # # Generate output
    # output = generator(messages)
    # generated_response = output[0]["generated_text"]
    # print(generated_response)


if __name__ == '__main__':
    main()
