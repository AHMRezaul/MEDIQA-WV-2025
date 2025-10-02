import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch



class Retriever:
    def __init__(self):
        # text-based retriever
        self.text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.text_index = faiss.read_index("retriever/index.faiss")
        with open("retriever/texts.json", "r", encoding="utf-8") as f:
            self.text_data = json.load(f)

        # image-based retriever
        self.image_index = faiss.read_index("retriever/image_index.faiss")
        with open("retriever/image_texts.json", "r", encoding="utf-8") as f:
            self.image_data = json.load(f)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def retrieve_by_text(self, query_text, top_k=2):
        vec = self.text_model.encode([query_text], normalize_embeddings=True).astype("float32")
        D, I = self.text_index.search(vec, top_k)
        return [self.text_data[i] for i in I[0]]

    def retrieve_by_image(self, image: Image.Image, top_k=2):
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vec = self.clip_model.get_image_features(**inputs).cpu().numpy()
        vec /= np.linalg.norm(vec)
        D, I = self.image_index.search(vec.astype("float32"), top_k)
        return [self.image_data[i]["source"] for i in I[0]]

    def retrieve_hybrid(self, query_text: str, query_image: Image.Image, top_k=2, alpha=0.5):
        # Encode text
        text_vec = self.text_model.encode([query_text], normalize_embeddings=True).astype("float32")
        D_text, I_text = self.text_index.search(text_vec, top_k * 2)

        # Encode image
        inputs = self.clip_processor(images=query_image, return_tensors="pt")
        with torch.no_grad():
            image_vec = self.clip_model.get_image_features(**inputs).cpu().numpy()
        image_vec /= np.linalg.norm(image_vec)
        D_img, I_img = self.image_index.search(image_vec.astype("float32"), top_k * 2)

        # Score map: encounter_id → combined score
        scores = {}

        for d, i in zip(D_text[0], I_text[0]):
            eid = self.text_data[i]["encounter_id"]
            scores[eid] = alpha * (1 - d)  # normalize to similarity

        for d, i in zip(D_img[0], I_img[0]):
            eid = self.image_data[i]["source"]["encounter_id"]
            scores[eid] = scores.get(eid, 0) + (1 - alpha) * (1 - d)

        # Sort and select
        sorted_eids = sorted(scores.items(), key=lambda x: -x[1])
        top_eids = {eid for eid, _ in sorted_eids[:top_k]}

        # Get full source entries
        return [entry["source"] for entry in self.image_data if entry["source"]["encounter_id"] in top_eids]

