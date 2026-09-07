<img width="858" height="279" alt="image" src="https://github.com/user-attachments/assets/eddcab1f-f6d1-4c59-a50e-930e6ec79261" />
# 📚 BookFinder - Book Recommendation RAG
BookFinder is a Retrieval-Augmented Generation (RAG) application that recommends books based on natural-language descriptions of what the user is looking for.
The project was developed as the final project for the **LLM Zoomcamp**, focusing on building a practical RAG pipeline from data ingestion and retrieval to reranking and answer generation.
The application uses a combination of **keyword search, semantic vector search and reranking** to retrieve relevant books before generating recommendations with an LLM.
---
## 🎯 Problem
Finding a book that matches a specific combination of interests can be difficult with traditional keyword-based search.
For example, a reader might ask:
> "I want a dark fantasy book with mystery and alchemy."
A simple keyword search may fail when the description uses different words to express the same concept.
BookFinder addresses this by combining lexical and semantic retrieval and then using a reranker to select the most relevant results before generating the final recommendation.
---
## 🏗️ Architecture
The current pipeline follows this flow:
```text
User Query
     │
     ▼
Keyword Search ──────┐
                     │
                     ▼
              Hybrid Search
              (RRF Fusion)
                     │
                     ▼
                Top Candidates
                     │
                     ▼
               Cross-Encoder
                 Reranking
                     │
                     ▼
               Top 5 Books
                     │
                     ▼
                  Groq LLM
                     │
                     ▼
             Book Recommendations
                     │
                     ▼
                Streamlit UI
```
### Retrieval
The project implements three retrieval approaches:
1. **Keyword Search** using MinSearch
2. **Vector Search** using ChromaDB and Sentence Transformers
3. **Hybrid Search** combining both approaches using Reciprocal Rank Fusion (RRF)
The hybrid approach combines lexical matching with semantic similarity.
### Reranking
The retrieved candidates are reranked using:
```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```
The reranker evaluates the relevance between the user's query and each retrieved book before the final context is sent to the LLM.
### Generation
The final answer is generated using a Groq-hosted LLM.
The prompt instructs the model to:
* recommend books from the retrieved context;
* provide the title and author;
* explain why each book matches;
* avoid inventing books or information;
* avoid recommending the exact book mentioned by the user.
---
## 📊 Dataset
The project uses a Google Books dataset containing book metadata such as:
* title
* author
* genre
* publication date
* description
* price
The dataset contains approximately 1,300 records.
Before indexing, missing values are handled and records without descriptions are removed.
The book information is then normalized into a common structure:
```python
{
    "id": "...",
    "title": "...",
    "author": "...",
    "genre": "...",
    "description": "...",
    "published_date": "..."
}
```
---
## 🔎 Retrieval
### Keyword Search
MinSearch indexes:
```text
title
author
genre
description
```
Different weights are applied to the fields, giving more importance to the description and genre when searching.
### Vector Search
Book metadata and descriptions are embedded using:
```text
all-MiniLM-L6-v2
```
The embeddings are stored in a persistent ChromaDB collection.
### Hybrid Search
The project combines keyword and vector results using **Reciprocal Rank Fusion**.
This allows the system to benefit from both:
* exact lexical matches;
* semantic similarity.
---
## 🧠 Reranking
After hybrid retrieval, the top candidates are passed through a Cross-Encoder reranker.
The pipeline therefore becomes:
```text
Query
  ↓
Keyword + Vector Retrieval
  ↓
Hybrid Results
  ↓
Cross-Encoder Reranking
  ↓
Top 5 Books
  ↓
LLM
```
This reduces the amount of irrelevant information passed to the generation model.
---
## 💬 Example
A user can enter a request such as:
```text
recommend fantasy books about mystery and alchemy
```
or:
```text
books like Lord of the Rings but darker
```
The system retrieves candidate books, reranks them and generates recommendations based on the retrieved context.
---
## 🖥️ Streamlit Interface
The project includes a Streamlit interface where users can enter their book preferences and receive recommendations.
### Application
The interface allows the user to:
1. enter a natural-language request;
2. retrieve relevant books;
3. receive an LLM-generated recommendation;
4. inspect the retrieved books and their descriptions.
> <img width="860" height="890" alt="image" src="https://github.com/user-attachments/assets/a62e5c5e-9525-4e0e-9028-9d4269384ae5" />
---
## 📈 Evaluation
The retrieval component is evaluated by comparing different retrieval strategies:
* Keyword Search
* Vector Search
* Hybrid Search
The evaluation uses:
* **Hit Rate@5**
* **MRR@5**
The goal is to determine which retrieval strategy provides the best results for the recommendation task.
### Retrieval Evaluation
<img width="278" height="160" alt="image" src="https://github.com/user-attachments/assets/157c586d-633c-46ce-87c1-2f7a9a5731ed" />
The final values should be updated with the actual results obtained when running the evaluation.
---
## 🛠️ Technologies
| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Application and data processing |
| Pandas                | Dataset processing              |
| Prefect               | Data Igestion                   |
| MinSearch             | Keyword retrieval               |
| Sentence Transformers | Text embeddings                 |
| ChromaDB              | Vector database                 |
| Cross-Encoder         | Reranking                       |
| Groq                  | LLM inference                   |
| Streamlit             | User interface                  |
---
## 📁 Project Structure
```text
bookfinder/
│
├── app.py
├── rag.py
├── notebook.ipynb
│
├── data_kaggle/
│   └── google_books_1299.csv
│
├── vectordb_books/
│
├── images/
│   ├── streamlit-home.png
│   └── retrieval-evaluation.png
│
├── requirements.txt
├── .env.example
└── README.md
```
---
## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/grbarbosa3/Bookfinder-Assistant
cd book_rag
```
Create a virtual environment:
```bash
python -m venv .venv
```
Activate it on Windows:
```bash
.venv\Scripts\activate
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
Create a `.env` file:
```env
GROQ_API_KEY=your_api_key_here
```
The API key should not be committed to the repository.
---
## ▶️ Running the Application
Start Streamlit with:
```bash
streamlit run app.py
```
The application will open in the browser.
---
## 🔬 Running the Retrieval Evaluation
The retrieval evaluation is available in the project notebook.
It compares:
```text
Keyword Search
Vector Search
Hybrid Search
```
using Hit Rate@5 and MRR@5.
Run the evaluation cells in:
```text
book_rag.ipynb
```
---
## 🚧 Current Limitations
The project is intentionally kept lightweight.
Current limitations include:
* the dataset is relatively small;
* the vector database is local;
* evaluation queries are generated from the dataset;
* LLM evaluation using an automated judge is not currently implemented;
* user feedback analytics are not currently implemented;
* the application is not containerized.
These are possible directions for future improvements.
---
## 🚀 Future Improvements
Possible extensions include:
* LLM-as-a-Judge evaluation;
* user feedback collection;
* analytics dashboard;
* automated ingestion pipeline;
* query rewriting;
* cloud deployment;
* containerization;
* larger and more diverse book datasets.
---
## 📌 Project Status
The core RAG application is functional and includes:
* [x] Dataset ingestion
* [x] Keyword retrieval
* [x] Vector retrieval
* [x] Hybrid retrieval
* [x] RRF fusion
* [x] Cross-Encoder reranking
* [x] LLM generation
* [x] Streamlit interface
* [x] Retrieval evaluation
* [ ] LLM-as-a-Judge
* [ ] Feedback system
* [ ] Analytics dashboard
* [ ] Automated orchestration
* [ ] Containerization
---
## 👨‍💻 Author

**Guilherme Rogério Barbosa**

Data Analyst transitioning into Data Engineering, with a background in Aeronautical Engineering and experience building data pipelines, analytical solutions and data applications.
Data Analyst & Data Engineering, with a background in Aeronautical Engineering and experience building data pipelines, analytical solutions and data applications.
