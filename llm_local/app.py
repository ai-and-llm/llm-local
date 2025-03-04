import json
import os
from json import JSONDecodeError

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM

from llm_local.embedding import EmbeddingModel
from llm_local.llm import LLMInstance, Llama, Phi
from llm_local.llm_with_tool import TemperatureFinderLLM


def login_to_huggingface() -> None:
    load_dotenv()
    hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
    login(hugging_face_token)


def chat_conversation() -> None:
    print("Chat started")

    llm = LLMInstance(Phi())
    while True:
        question = input("Your question: ")
        if question == "exit":
            break
        answer = llm.ask_question(question)
        print(f"Answer: {answer}")


def enhance_sentences() -> None:
    print("Make sentence more human readable")
    llm = LLMInstance(Llama())

    while True:
        sentence = input("Your sentence: ")
        if sentence == "exit":
            break
        enhanced_sentence = llm.enhance_sentence(sentence)
        print(f"ORIGINAL: {sentence}")
        print(f"ENHANCED: {enhanced_sentence}")


def movie_review_sentiment_analysis() -> None:
    print("Do sentiment analysis for the movie review")

    llm = LLMInstance(Llama())
    while True:
        movie_review_feedback = input("Write movie review: ")
        if movie_review_feedback == "exit":
            break

        review_result = llm.classify_movie_review(movie_review_feedback)

        print(f"SENTIMENT ANALYSIS: {review_result}")


def separate_words_embedding() -> None:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    model = AutoModel.from_pretrained("microsoft/deberta-base")
    tokens = tokenizer("Hello world", return_tensors="pt")
    for single_token in tokens["input_ids"][0]:
        print(tokenizer.decode(single_token))
    output = model(**tokens)[0]
    print(f"Model output: {output}")


def print_embedding(data: str) -> None:
    embedding = EmbeddingModel().generate_embedding(data)
    # Vector embedding (768): [ 0.02435045  0.02414671 -0.01585776]...[ 0.04622588 -0.02984796 -0.01665087]
    print(f"Vector embedding ({len(embedding)}): {embedding[:3]}...{embedding[-3:]}")


def main() -> None:
    login_to_huggingface()

    # chat_conversation()
    # enhance_sentences()
    # movie_review_sentiment_analysis()
    # print_embedding("Hello, world!!!")

    temp_finder = TemperatureFinderLLM()

    # response = temp_finder.find_temperature_at("Paris")
    # response = temp_finder.find_temperature_at("Madrid")
    response = temp_finder.find_temperature_at("San Francisco")


    print(response)


if __name__ == "__main__":
    main()
