
import os
import json

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ENVIRONMENT
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")


# STEP 1: LOAD JSONL

file_path = "gaali_data.jsonl"

documents = []
ids = []

with open(file_path, "r", encoding="utf-8-sig") as file:

    for line_number, line in enumerate(file, start=1):

        if not line.strip():
            continue

        try:
            item = json.loads(line)

            required_fields = [
                "id",
                "type",
                "trigger",
                "response",
                "intensity",
                "category"
            ]

            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Missing field '{field}'"
                    )

            content = f"""
            Type: {item['type']}
            Trigger words: {', '.join(item['trigger'])}
            Suggested response: {item['response']}
            Intensity: {item['intensity']}
            Category: {item['category']}
            """

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": item["type"],
                        "category": item["category"],
                        "intensity": int(item["intensity"])
                    }
                )
            )

            ids.append(item["id"])

        except Exception as e:

            raise ValueError(
                f"Error in JSONL line {line_number}: {e}"
            )


print("Total records loaded:", len(documents))

if not documents:
    raise ValueError("No valid records found!")


if len(ids) != len(set(ids)):
    raise ValueError("Duplicate IDs found in JSONL file!")


# STEP 2: EMBEDDINGS

embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)


# STEP 3: CHROMADB

vectorstore = Chroma(
    collection_name="hindi_gaali",
    persist_directory="gaali_db",
    embedding_function=embedding_model
)


# STEP 4: INSERT / UPDATE RECORDS

batch_size = 32

for start in range(0, len(documents), batch_size):

    batch_docs = documents[start:start + batch_size]
    batch_ids = ids[start:start + batch_size]

    vectorstore.add_documents(
        documents=batch_docs,
        ids=batch_ids
    )

    print(
        f"Processed {min(start + batch_size, len(documents))}"
        f"/{len(documents)}"
    )


# STEP 5: VERIFY

print("\n" + "=" * 40)
print("DATABASE CREATED SUCCESSFULLY")
print("=" * 40)

print("Records in JSONL:", len(documents))
print("Records in ChromaDB:", vectorstore._collection.count())
print("Collection: hindi_gaali")
print("Database folder: chroma_db")
