import os
import hashlib
import json
import openai
import pinecone
import re
from dotenv import load_dotenv
import hashlib


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
index = pinecone.Index("xgpt")


def generate_id(filename, chunk_index):
    """Generate a unique ID using the filename and chunk index."""
    return hashlib.sha256(f"{filename}_{chunk_index}".encode()).hexdigest()


def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def preprocess_text(text):
    """Replace all kinds of whitespaces with normal spaces."""
    return re.sub(r"\s", " ", text)


def chunk_text(text, chunk_size=2048):
    """Break text into chunks at sentence boundaries, up to `chunk_size` characters."""
    # Preprocess the text first to ensure all whitespaces are normal spaces
    text = preprocess_text(text)

    current_chunk = []
    current_chunk_size = 0
    sentence = []

    for token in text.split(" "):
        token_size = len(token) + 1  # Add 1 for the space
        sentence.append(token)
        current_chunk_size += token_size

        # Check for sentence boundary (".", "?", "!")
        is_sentence_boundary = token.endswith((".", "?", "!"))

        if current_chunk_size >= chunk_size and is_sentence_boundary:
            current_chunk.append(" ".join(sentence))
            yield " ".join(current_chunk)

            # Reset for next chunk
            current_chunk = []
            sentence = []
            current_chunk_size = 0
        elif is_sentence_boundary:
            current_chunk.append(" ".join(sentence))
            sentence = []

    # Output the remaining tokens as the last chunk
    if sentence:
        current_chunk.append(" ".join(sentence))
    if current_chunk:
        yield " ".join(current_chunk)


def main():
    output_folder = "outputs"
    chunks_folder = "chunks"

    # Create folder to store chunks if it doesn't exist
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)

    for filename in os.listdir(output_folder):
        if filename.endswith(".txt"):
            txt_path = os.path.join(output_folder, filename)

            with open(txt_path, "r") as f:
                text = f.read()

            for i, chunk in enumerate(chunk_text(text)):
                # Generate an embedding for each chunk
                embedding = get_embedding(chunk)

                # Generate a unique ID for each chunk
                chunk_id = generate_id(filename, i)

                # Upsert the embedding into Pinecone
                upsert_response = index.upsert(
                    vectors=[
                        (chunk_id, embedding, {"filename": filename, "chunk_index": i})
                    ]
                )

                print(upsert_response)

                # Save the chunk to a text file with the same ID
                chunk_file_path = os.path.join(chunks_folder, f"{chunk_id}.txt")
                with open(chunk_file_path, "w") as chunk_file:
                    chunk_file.write(chunk)


if __name__ == "__main__":
    main()
