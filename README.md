# Multimodal RAG System

A small, local Retrieval-Augmented Generation (RAG) system for multimodal documents (PDFs). This repository contains tools to extract text from PDFs, build a vector index, generate embeddings, and run a query/LLM pipeline for QA over the indexed content.

**Key goals**
- Extract text from PDFs and other multimodal sources
- Create embeddings and a vector store for retrieval
- Provide a simple LLM wrapper to answer questions using retrieved context
- Offer a notebook for training/experimentation

**Features**
- PDF extraction utilities
- Index building and storing vectors locally
- Modular embedding and LLM wrappers
- Example notebook for training and experiments

**Requirements**
- Python 3.10+ recommended
- A supported LLM API key (set in environment or config as required by `src/llm.py`)
- Install dependencies from `requirements.txt`

**Quick Setup (Windows PowerShell)**

```powershell
# create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

If you already have a preferred virtualenv manager, use that instead.

**Common Usage**

- Build an index from PDFs in `data/pdfs/` (example):

```powershell
python src\build_index.py --input_dir data\pdfs --output_dir data\index
```

- Run the main application (query interface):

```powershell
python src\main.py
```

- Inspect or run the training/experiment notebook:

Open `notebooks\multimodal-rag-training.ipynb` with Jupyter / VS Code Notebook.

Notes: Exact CLI options depend on the implementation in `src/*.py`. If a script expects different arguments, consult the script header or docstring.

**Repository Structure**

- `requirements.txt` — Python dependencies.
- `data/` — place source PDFs in `data/pdfs/` and store generated index files under `data/`.
- `notebooks/` — experimental notebooks (e.g., `multimodal-rag-training.ipynb`).
- `src/` — main project source files:
  - `build_index.py` — script to extract text from documents and build the vector index.
  - `embeddings.py` — embedding model wrapper (create embeddings for text chunks).
  - `extract.py` — PDF/document extraction utilities.
  - `llm.py` — wrapper around the language model used for generation.
  - `main.py` — entrypoint for running the query/assistant application.
  - `vector_store.py` — vector store and retrieval utilities.

**Environment & Configuration**

- If the project uses API keys (for embeddings or LLM), set them as environment variables before running the scripts. Example:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

- Alternatively, check the code for `.env` or config file support and populate accordingly.

**Troubleshooting**
- Missing packages: ensure the virtual environment is activated and `pip install -r requirements.txt` ran successfully.
- Errors when connecting to LLM/embeddings API: confirm API keys and network access.
- If index build fails on a PDF, try a different PDF or inspect `src/extract.py` for supported formats.


