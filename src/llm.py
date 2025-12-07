from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_llm(model_name: str = 'google/flan-t5-large') -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
    print(f'Loading LLM: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return tokenizer, model, device


def generate_answer_with_context(query: str, context_chunks: List[Dict], tokenizer, model, device: str) -> str:
    # Build a compact context
    context_text = "\n\n".join([
        f"Document: {c['metadata'].get('source','Unknown')}, Page {c['metadata'].get('page','N/A')}\n{c['content'][:400]}"
        for c in context_chunks[:3]
    ])

    prompt = f"Based on the following context from documents, answer the question accurately and concisely.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    try:
        inputs = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
        outputs = model.generate(inputs.input_ids, max_length=256, num_beams=4)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        sources = [f"{c['metadata'].get('source','Unknown')} (Page {c['metadata'].get('page','N/A')})" for c in context_chunks[:3]]
        if sources:
            answer += "\n\nSources: " + ', '.join(sources)
        return answer
    except Exception as e:
        return f'Error generating response: {e}'


def rag_query(query: str, vector_db, text_encoder, clip_processor, clip_model, tokenizer, model, device, k_text: int = 5, k_image: int = 2) -> Tuple[str, List[Dict], List[Dict]]:
    # Embed query for text
    query_emb = text_encoder.encode([query])
    text_results = vector_db.search_text_by_embedding(query_emb, k=k_text)

    # For images, use CLIP text encoder to get image-like embedding
    try:
        inputs = clip_processor(text=[query], return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        query_img_emb = text_features.cpu().numpy()
        image_results = vector_db.search_image_by_embedding(query_img_emb, k=k_image)
    except Exception:
        image_results = []

    answer = generate_answer_with_context(query, text_results, tokenizer, model, device)
    return answer, text_results, image_results
