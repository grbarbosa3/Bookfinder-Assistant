import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

from rag import ask_book_rag


st.set_page_config(
    page_title="BookFinder",
    page_icon="📚",
    layout="wide"
)



DB_PATH = "feedback.db"


def init_db():

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            answer TEXT,
            latency REAL,
            feedback INTEGER
        )
    """)

    conn.commit()
    conn.close()


init_db()


def save_feedback(
    query,
    answer,
    latency,
    feedback
):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO feedback
        (timestamp, query, answer, latency, feedback)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            query,
            answer,
            latency,
            feedback
        )
    )

    conn.commit()
    conn.close()



st.title("📚 BookFinder")

st.write(
    "Find book recommendations using "
    "hybrid search, vector search and reranking."
)


query = st.text_input(
    "What kind of book are you looking for?",
    placeholder="e.g. fantasy books darker than Lord of the Rings"
)


if st.button("Recommend books") and query:

    with st.spinner("Searching for books..."):

        result = ask_book_rag(query)

    st.session_state["result"] = result




if "result" in st.session_state:

    result = st.session_state["result"]

    st.subheader("Recommendations")

    st.write(result["answer"])

    st.caption(
        f"Response time: {result['latency']:.2f}s"
    )

    st.divider()

    st.subheader("Retrieved books")

    for book in result["books"]:

        with st.expander(book["title"]):

            st.write(
                f"**Author:** {book['author']}"
            )

            st.write(
                f"**Genre:** {book['genre']}"
            )

            st.write(
                book["description"]
            )

    st.divider()

    st.subheader("Was this recommendation useful?")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("👍 Yes"):

            save_feedback(
                result["query"],
                result["answer"],
                result["latency"],
                1
            )

            st.success(
                "Thanks for your feedback!"
            )

    with col2:

        if st.button("👎 No"):

            save_feedback(
                result["query"],
                result["answer"],
                result["latency"],
                0
            )

            st.info(
                "Thanks for the feedback."
            )