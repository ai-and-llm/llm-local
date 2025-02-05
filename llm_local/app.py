import os

from dotenv import load_dotenv
from huggingface_hub import login
from llm_local import llama, phi


def load_envs() -> None:
    load_dotenv()
    hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
    login(hugging_face_token)


def movie_review_sentiment_analysis(movie_review_feedback) -> None:
    print("Do sentiment analysis for the movie review")

    review_result = llama.classification(movie_review_feedback)

    print(f"REVIEW: {movie_review_feedback}")
    print(f"SENTIMENT ANALYSIS: {review_result}")


def enhance_sentence(sentence: str) -> None:
    print("Make sentence more human readable")

    # enhanced_sentence = llama.enhance_sentence(sentence)
    enhanced_sentence = phi.enhance_sentence(sentence)

    print(f"ORIGINAL: {sentence}")
    print(f"ENHANCED: {enhanced_sentence}")


def main() -> None:
    load_envs()

    movie_review_sentiment_analysis(
        "That was the most exciting horror movie that I've even seen."
    )
    # enhance_sentence("Questions to think through: do you need the request to complete in 5 sec or is it OK for the request to take some time?")


if __name__ == "__main__":
    main()
