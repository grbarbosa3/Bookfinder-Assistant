import os
import time

import pandas as pd
import chromadb

from dotenv import load_dotenv
from minsearch import Index
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions
from groq import Groq


DATA_PATH = "data_kaggle/google_books_1299.csv"
VECTOR_DB_PATH = "./vectordb_books"




df = pd.read_csv(DATA_PATH)
df = df.fillna("")
df = df[df["description"].str.strip() != ""]


books = []

for i, row in df.iterrows():

    genre = str(row["generes"])
    genre = genre.replace("&amp;", "&").replace(" ,", ",")

    books.append({
        "id": str(i),
        "title": str(row["title"]),
        "author": str(row["author"]),
        "genre": genre,
        "description": str(row["description"]),
        "published_date": str(row["published_date"])
    })



keyword_index = Index(
    text_fields=[
        "title",
        "author",
        "genre",
        "description"
    ]
)

keyword_index.fit(books)


def keyword_search(query, k=10):

    return keyword_index.search(
        query=query,
        boost_dict={
            "title": 1.0,
            "genre": 2.0,
            "description": 4.0,
            "author": 1.0
        },
        num_results=k
    )


embedding_function = (
    embedding_functions
    .SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

chroma_client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = chroma_client.get_or_create_collection(
    name="books",
    embedding_function=embedding_function
)


def vector_search(query, k=10):

    result = collection.query(
        query_texts=[query],
        n_results=k
    )

    results = []

    for i in range(len(result["ids"][0])):

        results.append({
            "id": result["ids"][0][i],
            "title": result["metadatas"][0][i]["title"],
            "author": result["metadatas"][0][i]["author"],
            "genre": result["metadatas"][0][i]["genre"],
            "description": result["documents"][0][i]
        })

    return results


def hybrid_search(query, k=10):

    keyword_results = keyword_search(query, k)
    vector_results = vector_search(query, k)

    scores = {}

    for rank, book in enumerate(keyword_results, 1):

        book_id = book["id"]

        scores[book_id] = (
            scores.get(book_id, 0)
            + 1 / (60 + rank)
        )

    for rank, book in enumerate(vector_results, 1):

        book_id = book["id"]

        scores[book_id] = (
            scores.get(book_id, 0)
            + 1 / (60 + rank)
        )

    book_map = {
        book["id"]: book
        for book in books
    }

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        book_map[book_id]
        for book_id in ranked_ids[:k]
    ]



reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, results):

    pairs = []

    for book in results:

        text = f"""
Title: {book['title']}
Genre: {book['genre']}
Description: {book['description']}
"""

        pairs.append(
            (query, text)
        )

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        book
        for book, score in ranked
    ]



load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found."
    )

groq_client = Groq(
    api_key=api_key
)


available_models = {
    model.id
    for model in groq_client.models.list().data
}


preferred_models = [
    "llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant"
]


model = next(
    (
        m
        for m in preferred_models
        if m in available_models
    ),
    None
)

if model is None:
    raise RuntimeError(
        "No compatible Groq model found."
    )


def ask_book_rag(query):

    start = time.time()

    candidates = hybrid_search(
        query,
        k=10
    )

    ranked_books = rerank(
        query,
        candidates
    )

    top_books = ranked_books[:5]

    context = "\n\n".join(
        f"""
Title: {book['title']}
Author: {book['author']}
Genre: {book['genre']}
Description: {book['description']}
"""
        for book in top_books
    )

    prompt = f"""
You are a book recommendation assistant.

User request:

{query}

Recommend books from the provided context
that best match the user's request.

For each recommendation:
- give the title
- give the author
- explain briefly why it matches

Use ONLY information from the context.

Do not invent books.
Do not invent information.

If the context contains books that are reasonably
related to the request, recommend them.

Do not recommend the exact book asked in the query.

Always give an indication even if it does not have
explicitly the exact words in query.

Context:
{context}
"""

    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    latency = time.time() - start

    return {
        "query": query,
        "answer": answer,
        "books": top_books,
        "model": model,
        "latency": latency
    }