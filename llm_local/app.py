import os

from dotenv import load_dotenv
from huggingface_hub import login
from llm_local import llama, phi


def load_envs() -> None:
    load_dotenv()
    hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
    login(hugging_face_token)


def main() -> None:
    load_envs()

    question = input("Make more human-readable: ")

    llama.chat(question)

    phi.chat(question)


if __name__ == "__main__":
    main()
