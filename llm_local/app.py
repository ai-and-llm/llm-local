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

    movie_review = input("Input movie review: ")

    review_result = llama.classification(movie_review)

    print(f"Review: {review_result}")

    # llama.chat(question)
    # phi.chat(question)


if __name__ == "__main__":
    main()
