import os
from typing import List, Dict
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_CACHE = os.path.join(ROOT, 'models_cache')
os.makedirs(MODEL_CACHE, exist_ok=True)


def load_text_encoder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    print(f'Loading text encoder: {model_name}')
    model = SentenceTransformer(model_name)
    return model


def load_clip(model_name: str = 'openai/clip-vit-base-patch32'):
    print(f'Loading CLIP model: {model_name}')
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    return clip_model, clip_processor


def generate_text_embeddings(texts: List[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    print(f'Generating text embeddings for {len(texts)} chunks...')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    return np.array(embeddings)


def generate_image_embeddings(image_data: List[Dict], clip_model, clip_processor, device: str = None) -> np.ndarray:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = clip_model.to(device)
    embeddings = []
    print(f'Generating image embeddings for {len(image_data)} images...')

    for img in image_data:
        try:
            inputs = clip_processor(images=img['image'], return_tensors='pt').to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            emb = image_features.cpu().numpy().reshape(-1)
            embeddings.append(emb)
        except Exception:
            # fallback zero vector (size 512 typical for CLIP ViT-B/32)
            embeddings.append(np.zeros(512))

    if len(embeddings) == 0:
        return np.zeros((0, 512))
    return np.vstack(embeddings)
