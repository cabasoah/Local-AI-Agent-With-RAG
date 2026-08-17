# Local AI Agent with RAG

A **fully local** Retrieval-Augmented Generation (RAG) system that leverages LangChain, Ollama, and Chroma to intelligently query and analyze structured datasets without external API dependencies. Built for analyzing restaurant reviews using semantic search and context-aware AI responses.

---

## 📋 Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Architecture & How It Works](#architecture--how-it-works)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## ✨ Features

- **Local-First Architecture**: No external API calls or cloud dependencies
- **Semantic Search**: Retrieves contextually relevant reviews using vector similarity (Cosine Similarity)
- **Context-Aware Responses**: Injects retrieved reviews into prompts for grounded AI-generated answers
- **Persistent Vector Store**: Chroma database caches embeddings for fast subsequent queries
- **Interactive CLI**: User-friendly command-line interface for querying the dataset
- **Optimized for Small Datasets**: Perfect for domain-specific RAG applications
- **Extensible Design**: Easy to adapt for other structured datasets

---

## 🛠 Technology Stack

| Component | Purpose | Details |
|-----------|---------|---------|
| **LangChain** | Orchestration Framework | Chains LLM calls, prompts, and retrieval logic |
| **Ollama** | Local LLM Runtime | Runs Llama 3.2 (chat) and mxbai-embed-large (embeddings) |
| **Chroma** | Vector Database | Stores and retrieves vector embeddings efficiently |
| **Pandas** | Data Processing | Loads and processes CSV datasets |
| **Python 3.8+** | Runtime | Core language |

---

## 📦 Prerequisites

Before installation, ensure you have:

1. **Python 3.8 or higher** - [Download](https://www.python.org/)
2. **Ollama** - [Install from ollama.com](https://ollama.com)
3. **Ollama Models** (download via Ollama):
   ```bash
   ollama pull llama3.2      # LLM for generating responses
   ollama pull mxbai-embed-large  # Embedding model
   ```
4. **Git** (optional) - for cloning the repository

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Local-AI-Agent-With-RAG.git
cd Local-AI-Agent-With-RAG
```

### Step 2: Create a Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Ensure Ollama is Running

Start the Ollama service:

```bash
# On macOS/Linux
ollama serve

# On Windows (if installed)
ollama serve
```

Verify models are installed:

```bash
ollama list
```

You should see:
- `llama3.2`
- `mxbai-embed-large`

---

## 📁 Project Structure

```
Local-AI-Agent-With-RAG/
├── main.py                            # Interactive CLI for querying
├── vector.py                          # ETL & Vector embedding pipeline
├── realistic_restaurant_reviews.csv   # Sample dataset
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── chrome_langchain_db/               # Chroma vector store (auto-generated)
    └── [vector embeddings cache]
```

### File Descriptions

| File | Purpose |
|------|---------|
| `vector.py` | Loads CSV, creates embeddings, stores in Chroma DB |
| `main.py` | Runs interactive query loop with RAG pipeline |
| `realistic_restaurant_reviews.csv` | Sample dataset (restaurant reviews) |
| `requirements.txt` | Python package dependencies |

---

## 🎯 Usage

### Quick Start

1. **Initialize the Vector Store** (run once):
   ```bash
   python vector.py
   ```
   This will:
   - Load reviews from `realistic_restaurant_reviews.csv`
   - Generate embeddings using `mxbai-embed-large`
   - Store them in `./chrome_langchain_db/`

2. **Start Querying**:
   ```bash
   python main.py
   ```
   Then enter questions about the restaurant reviews:
   ```
   Ask your question (q to quit): What do customers say about the pizza quality?

   [System retrieves top 5 relevant reviews and generates an answer using Llama 3.2]
   ```

3. **Exit**:
   Type `q` and press Enter to quit.

### Example Queries

- "What are the most common complaints?"
- "How do customers rate the service?"
- "What makes this restaurant special?"
- "Are there any recurring themes in the reviews?"

---

## 🏗 Architecture & How It Works

### Overview: Two-Stage Pipeline

```
CSV Data → Vector Embedding → Chroma Vector DB → User Query → Semantic Search → Context Injection → LLM → Answer
```

### Stage 1: Embedding Pipeline (`vector.py`)

1. **Data Loading**: Reads `realistic_restaurant_reviews.csv` with Pandas
2. **Document Creation**: Converts each review row into LangChain `Document` objects
   - `page_content`: Title + Review text
   - `metadata`: Rating, Date
   - `id`: Row index
3. **Embedding**: Uses Ollama's `mxbai-embed-large` to generate vector embeddings
4. **Storage**: Persists embeddings in Chroma's local database (`./chrome_langchain_db/`)

**Run once per dataset change:**
```python
python vector.py
```

### Stage 2: Query Pipeline (`main.py`)

1. **User Input**: Accepts natural language questions
2. **Query Embedding**: Converts question into a vector using the same embedding model
3. **Semantic Retrieval**: Finds top 5 most similar reviews using Cosine Similarity
4. **Context Injection**: Embeds retrieved reviews into a prompt template
5. **LLM Generation**: Llama 3.2 generates grounded answer based on context
6. **Output**: Returns AI-generated response

**Prompt Template:**
```
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: [TOP 5 REVIEWS]

Here is the question to answer: [USER QUESTION]
```

### Why Chroma over PostgreSQL/pgvector?

- **Minimal setup**: No database server configuration
- **Fast for small datasets**: Optimized for embedded use cases
- **Persistent storage**: Embeddings cached locally
- **Zero external dependencies**: Fully self-contained

---

## ⚙️ Configuration

### Customizing the System

Edit `main.py` or `vector.py` to modify:

#### 1. **Embedding Model** (`vector.py`):
```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # Change model here
```

#### 2. **LLM Model** (`main.py`):
```python
model = OllamaLLM(model="llama3.2")  # Change to another Ollama model
```

#### 3. **Number of Retrieved Reviews** (`vector.py`):
```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Change "k" to adjust number of reviews
)
```

#### 4. **Prompt Template** (`main.py`):
```python
template = """
You are an expert in answering questions about a pizza restaurant
...
"""
```

#### 5. **Data Source**:
Replace `realistic_restaurant_reviews.csv` with your own CSV file.
Ensure columns include at least: `Title`, `Review`, `Rating`, `Date`

---

## 🐛 Troubleshooting

### Issue: `ConnectionRefusedError` when connecting to Ollama

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Issue: Model not found error

**Solution**: Download required models:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Issue: CSV file not found

**Solution**: Ensure `realistic_restaurant_reviews.csv` is in the project root directory.

### Issue: Slow response time on first query

**Solution**: This is normal—Ollama is loading the model into memory. Subsequent queries will be faster.

### Issue: Vector store not updating

**Solution**: Delete the `chrome_langchain_db/` directory and re-run `vector.py`:
```bash
rm -r chrome_langchain_db  # On Windows: rmdir chrome_langchain_db /s
python vector.py
```

---

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Chroma Documentation](https://docs.trychroma.com/)
- [RAG Fundamentals](https://www.promptingguide.ai/techniques/rag)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Submit pull requests
- Suggest improvements
- Add support for additional data formats

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 💡 Notes

- **Privacy**: All data stays local—no external API calls or cloud storage
- **Performance**: Vector embeddings are cached; subsequent queries are fast
- **Scalability**: Designed for datasets up to thousands of documents; for larger scales, consider production RAG systems
- **Customization**: Easily adaptable to other domains by changing the dataset and prompt template

---

## 🎓 What You'll Learn

By exploring this project, you'll understand:
- How RAG systems work end-to-end
- Embedding generation and semantic search
- LLM prompt engineering and chaining
- Vector database usage (Chroma)
- Building local AI applications

**Happy querying! 🚀**
