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


def query_text(text, message_history):
    embedding = get_embedding(text + message_history[-1]["content"])

    query_response = index.query(
        top_k=5,
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

    messages = [
        {"role": "system", "content": "You are a scientist at a national lab."},
    ]
    messages = messages + message_history[-5:]
    messages.append(
        {
            "role": "assistant",
            "content": f"this is additional context: {combined_context}",
        },
    )

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
        messages.append({"role": "user", "content": text_input})
        text_response = query_text(text_input, messages)
        messages.append({"role": "assistant", "content": text_response})
