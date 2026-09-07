class RAG:

    def __init__(
        self,
        keyword_index,
        collection,
        reranker,
        groq_client,
        model
    ):
        self.keyword_index = keyword_index
        self.collection = collection
        self.reranker = reranker
        self.groq_client = groq_client
        self.model = model


    def search(self, query, k=10):

        keyword_results = keyword_search(query, k)
        vector_results = vector_search(query, k)

        scores = {}

        # Keyword ranking
        for rank, book in enumerate(keyword_results, 1):

            book_id = book["id"]

            scores[book_id] = (
                scores.get(book_id, 0)
                + 1 / (60 + rank)
            )

        # Vector ranking
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


    def rerank_results(self, query, results):

        pairs = []

        for book in results:

            text = f"""
Title: {book['title']}
Genre: {book['genre']}
Description: {book['description']}
"""

            pairs.append((query, text))

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            book
            for book, score in ranked
        ]


    def generate_answer(self, query, books):

        context = "\n\n".join(
            f"""
Title: {book['title']}
Author: {book['author']}
Genre: {book['genre']}
Description: {book['description']}
"""
            for book in books
        )

        prompt = f"""
You are a book recommendation assistant.

User request:

{query}

Recommend books from the context that best
match the user's request.

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

Always provide recommendations when there are
reasonably related books, even if they do not
contain the exact words from the query.

Context:

{context}
"""

        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content


    def ask(self, query):

        start = time.time()

        # 1. Hybrid retrieval
        candidates = self.search(
            query,
            k=10
        )

        # 2. Reranking
        ranked_books = self.rerank_results(
            query,
            candidates
        )

        # 3. Top 5
        top_books = ranked_books[:5]

        # 4. Generate answer
        answer = self.generate_answer(
            query,
            top_books
        )

        latency = time.time() - start

        return {
            "query": query,
            "answer": answer,
            "books": top_books,
            "model": self.model,
            "latency": latency
        }