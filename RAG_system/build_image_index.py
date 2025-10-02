from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import json
import numpy as np
import os
import torch

IMG_DIR = "dataset/mediqa-wv/images"
TRAIN_FILE = "dataset/mediqa-wv/json_files/train-valid.json"
INDEX_OUT = "retriever/image_index.faiss"
METADATA_OUT = "retriever/image_texts.json"



clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# clip_model = CLIPModel.from_pretrained(
#     "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#     trust_remote_code=True
# )
# clip_processor = CLIPProcessor.from_pretrained(
#     "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#     trust_remote_code=True
# )

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

image_vecs = []
image_meta = []

for ex in data:
    for fn in ex["image_ids"]:
        path = os.path.join(IMG_DIR, fn)
        if not os.path.exists(path): continue

        try:
            image = Image.open(path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs).cpu().numpy()[0]
            emb /= np.linalg.norm(emb)  # normalize
            image_vecs.append(emb)
            image_meta.append({
                "image_id": fn,
                "source": ex
            })
        except Exception as e:
            print(f"[WARN] Failed on {fn}: {e}")

if not image_vecs:
    raise ValueError("No image embeddings generated. Check if your images exist and can be read.")

dim = image_vecs[0].shape[0]
index = faiss.IndexFlatIP(dim)

index.add(np.array(image_vecs).astype("float32"))
os.makedirs(os.path.dirname(INDEX_OUT), exist_ok=True)
faiss.write_index(index, INDEX_OUT)

with open(METADATA_OUT, "w", encoding="utf-8") as f:
    json.dump(image_meta, f, ensure_ascii=False, indent=2)

print(f"[+] Saved image FAISS index and metadata.")
