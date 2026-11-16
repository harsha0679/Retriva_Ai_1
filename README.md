# Retriva-AI 🧠

**Retriva-AI** is an advanced AI-powered knowledge platform that allows you to chat with your documents. Think of it as a super-smart librarian: you hand it your company policies, resumes, or manuals, and you can ask it any question. It instantly finds the exact pages and gives you a direct, accurate answer.

Everything runs **100% locally** and securely on your machine, meaning your private documents are never sent to external servers like OpenAI.

---

## 🚀 How It Works (The Architectural Flow)

We use a technique called **RAG (Retrieval-Augmented Generation)**. Here is the simple step-by-step flow of what happens when you ask a question:

1. **You Ask a Question:** You type a question in the chat (e.g., *"What is the policy for sick leave?"*).
2. **Query Expansion (The Translation):** Sometimes people ask messy questions. Retriva-AI silently asks the AI to rewrite your question into a formal, highly specific search query to ensure it doesn't miss anything.
3. **Hybrid Search (The Librarian):** The system searches your uploaded documents using two methods simultaneously:
   - *Vector Search:* Looks for the "meaning" of your question.
   - *Keyword Search:* Looks for exact word matches (BM25).
4. **Cross-Encoder Re-Ranking (The Filter):** The system pulls the top 30 paragraphs that might have the answer. A specialized AI "Re-ranker" reads all 30 paragraphs and picks the absolute top 5 best matches.
5. **The Final Answer (The Brain):** The top 5 paragraphs are sent to the Large Language Model (Qwen 2.5). The AI reads them, finds the exact answer, and generates a polite, clear response back to you.

---

## 💬 Simple Query Example

**User:** *"What is the name of the company in this handbook?"*

**Behind the Scenes:** 
- *Step 1:* The system pulls the document title from the metadata (Contextual Chunking).
- *Step 2:* It retrieves the first page of the handbook.
- *Step 3:* The LLM reads the first page and formulates the response.

**Retriva-AI:** *"The name of the company mentioned in the handbook is TechCorp Inc."*

---

## 🛠️ Tech Stack & Methods Used

This project uses cutting-edge Generative AI tools optimized for local performance:

### The Interface
- **Frontend:** React, TypeScript, Vite
- **UI Design:** Glassmorphism, TailwindCSS, Framer Motion (Animations)

### The Backend (FastAPI & Python)
- **Framework:** FastAPI (High-performance API routing)
- **Vector Database:** ChromaDB (Stores document vectors locally)
- **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`)
- **Large Language Model (LLM):** Ollama running `Qwen 2.5 (7B parameters)`

### Advanced RAG Methods Implemented
- **Hybrid Search:** Fusing Dense Vectors and Sparse BM25 keywords for maximum search accuracy.
- **Contextual Chunking:** Automatically injecting document titles and summaries into every paragraph so the AI never loses context.
- **Query Expansion:** Using the LLM to rewrite and optimize user questions before searching.
- **Cross-Encoder Re-Ranking:** Using a dedicated reranker model (`bge-reranker-base`) to sort search results by true semantic relevance.

---

## ⚙️ How to Run Locally

### Prerequisites
1. Install [Python 3.10+](https://www.python.org/)
2. Install [Node.js](https://nodejs.org/)
3. Install [Ollama](https://ollama.com/)

### 1. Start the Backend
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (Ollama will auto-start in the background!)
uvicorn main:app --port 8080 --reload --reload-dir app
```

### 2. Start the Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` in your browser and start chatting with your documents!
