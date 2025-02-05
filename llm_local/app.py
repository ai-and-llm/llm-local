import os

from dotenv import load_dotenv
from huggingface_hub import login
from llm_local.chat import Chat, Llama


def load_envs() -> None:
    load_dotenv()
    hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
    login(hugging_face_token)


# chat = Chat(Phi())
chat = Chat(Llama())


def chat_conversation() -> None:
    print("Chat started")

    while True:
        question = input("Your question: ")
        if question == "exit":
            break
        answer = chat.ask_question(question)
        print(f"Answer: {answer}")


def enhance_sentences() -> None:
    print("Make sentence more human readable")

    while True:
        sentence = input("Your sentence: ")
        if sentence == "exit":
            break
        enhanced_sentence = chat.enhance_sentence(sentence)
        print(f"ORIGINAL: {sentence}")
        print(f"ENHANCED: {enhanced_sentence}")


def movie_review_sentiment_analysis() -> None:
    print("Do sentiment analysis for the movie review")

    while True:
        movie_review_feedback = input("Write movie review: ")
        if movie_review_feedback == "exit":
            break

        review_result = chat.classify_movie_review(movie_review_feedback)

        print(f"SENTIMENT ANALYSIS: {review_result}")


def main() -> None:
    load_envs()

    # chat_conversation()

    # enhance_sentences()

    movie_review_sentiment_analysis()


if __name__ == "__main__":
    main()
