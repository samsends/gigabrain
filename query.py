import os
import pinecone
from dotenv import load_dotenv
import openai
from embed import get_embedding
from colorama import Fore


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
index = pinecone.Index("xgpt")


def query_text(text, previous_messages):
    embedding_text = text
    if len(previous_messages) > 2:
        embedding_text = embedding_text + previous_messages[-2]["content"]
    embedding = get_embedding(embedding_text)

    query_response = index.query(
        top_k=10,
        include_values=True,
        include_metadata=True,
        vector=embedding,
    )

    chunks_folder = "chunks"
    context = []

    for result in query_response["matches"]:
        chunk_id = result["id"]
        chunk_file_path = os.path.join(chunks_folder, f"{chunk_id}.txt")

        with open(chunk_file_path, "r") as chunk_file:
            chunk_text = chunk_file.read()

        context.append(chunk_text)

    combined_context = "\n\n".join(context)

    # system_message = [
    #     {
    #         "role": "system",
    #         "content": "You are not only a brilliant scientist at the forefront of your field, but you also excel at breaking down complex ideas into digestible pieces, much like Richard Feynman or Neil deGrasse Tyson. Your passion for educating others elevates your work, making you both a researcher and an exceptional communicator.",
    #     },
    # ]
    system_message = [
        {
            "role": "system",
            "content": "You are a brilliant scientist at the forefront of your field and excellent communicator.",
        },
    ]

    messages = [
        {
            "role": "user",
            "content": f"Respond to the following: {text}\n\n\nHere are some optional research snippets that you can use to enhance your answer, but re-explain the snippet if used because the reader has no access to it. : {combined_context}",
        },
    ]
    messages = system_message + previous_messages[-5:] + messages

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=messages, stream=True
    )
    collected_chunks = []
    for chunk in response:
        if chunk["choices"][0]["delta"]:
            chunk_message = chunk["choices"][0]["delta"][
                "content"
            ]  # extract the message
            collected_chunks.append(chunk_message)  # save the message
            print(Fore.CYAN + f"{chunk_message}", end="", flush=True)
    return "".join(collected_chunks)


if __name__ == "__main__":
    messages = []

    while True:
        print("\n")
        text_input = input(Fore.WHITE + "Enter text to query: ")
        print("\n")
        text_response = query_text(text_input, messages)
        messages.append({"role": "assistant", "content": text_response})
