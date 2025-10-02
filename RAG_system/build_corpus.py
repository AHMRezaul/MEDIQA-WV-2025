import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

TRAIN_FILE = "dataset/mediqa-wv/json_files/train-valid.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "retriever/index.faiss"
TEXTS_FILE = "retriever/texts.json"

def main():
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = []
    vectors = []

    for ex in data:
        query = f"{ex['query_title_en']}  {ex['query_content_en']}"
        texts.append(ex)
        emb = model.encode(query, normalize_embeddings=True)
        vectors.append(emb)

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print(f"[+] Saved FAISS index and metadata with {len(texts)} examples.")

if __name__ == "__main__":
    main()
