"""Simple helper to extract text from PDFs and save chunks + metadata.

Run:
    python src/build_index.py

This script is a minimal template — expand with embeddings, image extraction,
OCR, and FAISS index building as needed.
"""
import os
import fitz  # PyMuPDF
import pickle

ROOT = os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else '.'
PDF_DIR = os.path.join(ROOT, 'data', 'pdfs')
OUT_DIR = os.path.join(ROOT, 'embeddings')
os.makedirs(OUT_DIR, exist_ok=True)

text_chunks = []
metadata = []

pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]

if not pdf_files:
    print('No PDF files found in:', PDF_DIR)
    print('Place your PDFs in the project folder `data/pdfs/` then re-run.')
    raise SystemExit(1)

for pdf_path in pdf_files:
    print('Processing', pdf_path)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text and text.strip():
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            for para in paragraphs:
                text_chunks.append(para)
                metadata.append({'source': os.path.basename(pdf_path), 'page': page_num+1})
    doc.close()

# Save extracted chunks + metadata for later embedding/indexing
with open(os.path.join(OUT_DIR, 'text_chunks.pkl'), 'wb') as f:
    pickle.dump({'texts': text_chunks, 'metadata': metadata}, f)

print(f"Saved {len(text_chunks)} text chunks to {OUT_DIR}/text_chunks.pkl")
