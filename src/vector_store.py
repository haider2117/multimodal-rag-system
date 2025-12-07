import numpy as np
import faiss
from typing import List, Dict, Tuple


class VectorDatabase:
    """Simple FAISS-backed vector store for text and image embeddings."""

    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.text_chunks: List[str] = []
        self.image_chunks: List[Dict] = []
        self.metadata: List[Dict] = []

    def build_indexes(self, text_emb: np.ndarray, image_emb: np.ndarray, texts: List[str], images: List[Dict], meta: List[Dict]):
        # build text index
        if text_emb is None or text_emb.size == 0:
            self.text_index = None
        else:
            dim_t = text_emb.shape[1]
            self.text_index = faiss.IndexFlatL2(dim_t)
            self.text_index.add(text_emb.astype('float32'))

        # build image index
        if image_emb is None or image_emb.size == 0:
            self.image_index = None
        else:
            dim_i = image_emb.shape[1]
            self.image_index = faiss.IndexFlatL2(dim_i)
            self.image_index.add(image_emb.astype('float32'))

        self.text_chunks = texts
        self.image_chunks = images
        self.metadata = meta

        print(f'Built indexes. Texts: {len(texts)}, Images: {len(images)}')

    def search_text_by_embedding(self, query_emb: np.ndarray, k: int = 5) -> List[Dict]:
        if self.text_index is None:
            return []
        distances, indices = self.text_index.search(query_emb.astype('float32'), k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                results.append({'content': self.text_chunks[idx], 'metadata': self.metadata[idx], 'score': float(1 / (1 + dist))})
        return results

    def search_image_by_embedding(self, query_emb: np.ndarray, k: int = 3) -> List[Dict]:
        if self.image_index is None:
            return []
        distances, indices = self.image_index.search(query_emb.astype('float32'), k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.image_chunks):
                results.append({'image': self.image_chunks[idx], 'metadata': self.metadata[idx], 'score': float(1 / (1 + dist))})
        return results
