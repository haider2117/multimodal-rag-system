import os
import argparse
import pickle

from extract import DocumentProcessor
from embeddings import load_text_encoder, load_clip, generate_text_embeddings, generate_image_embeddings
from vector_store import VectorDatabase
from llm import load_llm, rag_query

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PDF_DIR = os.path.join(ROOT, 'data', 'pdfs')
EMB_DIR = os.path.join(ROOT, 'embeddings')
os.makedirs(EMB_DIR, exist_ok=True)


def find_pdfs() -> list:
    if not os.path.exists(PDF_DIR):
        return []
    return [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]


def build_pipeline(rebuild_embeddings: bool = False):
    # 1. Extract
    pdf_files = find_pdfs()
    if not pdf_files:
        raise SystemExit('No PDF files found in data/pdfs. Please add PDFs and re-run.')

    proc = DocumentProcessor()
    texts, images, meta = proc.extract_from_pdfs(pdf_files)

    # 2. Load encoders
    text_encoder = load_text_encoder()
    clip_model, clip_processor = load_clip()

    # 3. Generate or load embeddings
    text_emb_path = os.path.join(EMB_DIR, 'text_embeddings.npy')
    image_emb_path = os.path.join(EMB_DIR, 'image_embeddings.npy')
    meta_path = os.path.join(EMB_DIR, 'metadata.pkl')

    if not rebuild_embeddings and os.path.exists(text_emb_path) and os.path.exists(image_emb_path) and os.path.exists(meta_path):
        print('Loading saved embeddings...')
        import numpy as np
        text_emb = np.load(text_emb_path)
        image_emb = np.load(image_emb_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        text_emb = generate_text_embeddings(texts, text_encoder)
        image_emb = generate_image_embeddings(images, clip_model, clip_processor)
        import numpy as np
        np.save(text_emb_path, text_emb)
        np.save(image_emb_path, image_emb)
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)

    # 4. Build vector DB
    vector_db = VectorDatabase()
    vector_db.build_indexes(text_emb, image_emb, texts, images, meta)

    # 5. Load LLM
    tokenizer, llm_model, device = load_llm()

    return {
        'vector_db': vector_db,
        'text_encoder': text_encoder,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'tokenizer': tokenizer,
        'llm_model': llm_model,
        'device': device
    }


def main():
    parser = argparse.ArgumentParser(description='Run Multimodal RAG pipeline')
    parser.add_argument('--query', type=str, default=None, help='Query to run after building the pipeline')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild embeddings even if saved ones exist')

    args = parser.parse_args()

    env = build_pipeline(rebuild_embeddings=args.rebuild)

    if args.query:
        answer, texts, images = rag_query(args.query, env['vector_db'], env['text_encoder'], env['clip_processor'], env['clip_model'], env['tokenizer'], env['llm_model'], env['device'])
        print('\n=== ANSWER ===\n')
        print(answer)
        print('\n=== RETRIEVED TEXTS ===\n')
        for t in texts:
            print(f"- {t['metadata'].get('source','')} (page {t['metadata'].get('page','N/A')}): {t['content'][:200]}...\n")

    else:
        print('\nPipeline built successfully. Run with `--query "Your question"` to ask a question.')


if __name__ == '__main__':
    main()
