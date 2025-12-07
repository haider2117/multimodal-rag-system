# Multimodal RAG — Notebook Project

This repository contains a single Jupyter notebook implementing a multimodal
Retrieval-Augmented Generation (RAG) demo over PDF documents. It's packaged
so you can upload it to GitHub as an AI/ML project for your resume.

Contents in this repository (cleaned):

- `multimodal-rag-training.ipynb` — Main Jupyter notebook (analysis, extraction, embeddings,
	retrieval, and Gradio demo).
- `data/pdfs/` — Place your PDF dataset here (currently contains 3 PDFs).
- `src/build_index.py` — Helper script to extract text chunks and save them
	for embedding/indexing.
- `requirements.txt` — Python packages required to run the notebook.
- `.gitignore` — Ignores temp files, caches, and embeddings.
- `prompts.txt` — (optional) prompt examples you used while developing.

What I changed for a clean GitHub upload:

- Removed the `report.pdf` (assignment report) to avoid unwanted impressions.
- Removed empty folders (`temp/`, `models_cache/`, `embeddings/`, `scripts/`).
- Moved the helper script into `src/` and updated run instructions.
- Updated this `README.md` to be concise and GitHub-friendly.

Project structure (final):

```
reg/                     # repo root
├── multimodal-rag-training.ipynb
├── data/
│   └── pdfs/            
├── src/
│   └── build_index.py
├── requirements.txt
├── .gitignore
├── prompts.txt
└── README.md
```

Quick setup (PowerShell)
```powershell
# create local folders (if not present)
mkdir data\pdfs

# install dependencies
python -m pip install -r requirements.txt

# run the helper to extract text chunks (saves to embeddings/ by default)
python src\build_index.py

# start Jupyter and open the notebook
jupyter notebook multimodal-rag-training.ipynb
```

Windows-specific notes
- Tesseract OCR: install from the official releases and add to PATH, or set in
	the notebook:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```
- Poppler (for `pdf2image`): download a Windows build and either add
	`poppler/bin` to PATH or pass `poppler_path` to `convert_from_path()`.
- PyTorch: install the correct wheel for your CUDA version or CPU-only wheel
	(see https://pytorch.org).

Notebook adjustments you should make before publishing
- Remove Colab-specific code (the notebook currently includes a cell using
	`google.colab.files.upload()`). Replace with a local loader:
```python
import os
pdf_files = [os.path.join('data','pdfs', f) for f in os.listdir('data/pdfs') if f.lower().endswith('.pdf')]
```
- Change `demo.launch(share=True)` to `demo.launch(share=False)` to avoid
	exposing a public share link when running locally.

Tips for a good GitHub project / resume entry
- Add a short `DESCRIPTION` on the repo front page summarizing the problem,
	approach, and results (1–3 sentences).
- Add a screenshot of the Gradio UI and an example query/answer in the
	repository `README.md` (optional but recommended).
- Include a short `CONTRIBUTION` or `USAGE` section listing how to reproduce
	your main results (we included run commands above).

If you want, I can now:
- Patch `multimodal-rag-training.ipynb` to remove the Colab upload cell and use the local
	`data/pdfs/` loader automatically, and add the Tesseract/poppler hints inline.
- Create a `README` screenshot and short description tailored for your resume.

Which would you like me to do next?
---

## **Demo (Notebook)**

- The interactive notebook was moved into `notebooks/multimodal-rag-training.ipynb`.

If you want an HTML snapshot or screenshots for the README, generate them locally and add them to the repo; they were intentionally not included here.
