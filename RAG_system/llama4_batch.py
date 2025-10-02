# llama4_batch_routeA.py (finalized, cleaned, debugged)

import os
import json
from typing import List, Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from retriever import Retriever


MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
HF_TOKEN = "your_hf_token" # replace with your actual Hugging Face token
CACHE_DIR = "../.cache"

TEST_FILE     = "dataset/mediqa2025-wv-testinputs/test_inputonly.json"
# FEWSHOT_FILE  = "dataset/mediqa-wv/json_files/fewshot.json"
IMAGES_ROOT   = "dataset/mediqa2025-wv-testinputs/images_test"
DICT_FILE = "dataset/mediqa-wv/json_files/data_dictionary.txt"
OUTPUT_FILE   = "RAG_system/result/predictions.json"
RAW_OUTPUT_FILE = "RAG_system/result/raw_generations.txt"


TARGET_SIZE = (224, 224)


# === Helpers ===
def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def parse_metadata_dict(path: str) -> Dict[str, List[str]]:
    categories = {}
    current_key = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#-") or line.startswith("#reference"):
                continue
            if line.startswith("#"):
                current_key = line[1:].strip()
                categories[current_key] = []
            else:
                categories[current_key].append(line.strip())
    return categories

def make_system_prompt(metadata_dict: Dict[str, List[str]]) -> str:
    lines = []
    for key, values in metadata_dict.items():
        joined = ", ".join(values)
        lines.append(f"- {key}: [{joined}]")
    return (
        "You are a bilingual wound-care assistant.\n"
        "Your response must be a valid JSON with exactly two keys: 'metadata' and 'responses'.\n"
        "'metadata' includes the following fields and must use only one of the allowed values for each:\n" +
        "\n".join(lines) + "\n" +
        "'responses' must be concise instructions under 120 words. Not generic solutions.\n"
        "Do not add new fields or invent new labels. Only use allowed metadata values."
    )


def build_chat_and_images(
    system_msg: str,
    exemplars: List[Dict],
    case: Dict,
    img_root: str,
) -> Tuple[List[Dict], List[Image.Image]]:
    def txt(text: str) -> Dict:
        return {"type": "text", "text": text}

    def load_and_resize_image(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB").resize(TARGET_SIZE)

    msgs: List[Dict] = [{"role": "system", "content": [txt(system_msg)]}]
    images: List[Image.Image] = []

    for ex in exemplars:
        ex_imgs = []
        for fn in ex.get("image_ids", []):
            img_path = os.path.join(img_root, fn)
            if os.path.exists(img_path):
                try:
                    img = load_and_resize_image(img_path)
                    ex_imgs.append(img)
                except Exception as e:
                    print(f"[WARN] Could not open exemplar image {fn}: {e}")
        images.extend(ex_imgs)

        user_parts = ([{"type": "image", "image": img} for img in ex_imgs] +
                      [txt(ex["query_content_en"])])
        msgs.append({"role": "user", "content": user_parts})

        if ex.get("responses"):
            msgs.append({
                "role": "assistant",
                "content": [txt(ex["responses"][0].get("content_en", ""))]
            })

    cur_imgs = []
    for fn in case.get("image_ids", []):
        img_path = os.path.join(img_root, fn)
        if os.path.exists(img_path):
            try:
                img = load_and_resize_image(img_path)
                cur_imgs.append(img)
            except Exception as e:
                print(f"[WARN] Could not open case image {fn}: {e}")
        else:
            print(f"[WARN] Case image {fn} not found.")
    images.extend(cur_imgs)

    question_en = f"{case['query_title_en']}  {case['query_content_en']}"
    user_parts = ([{"type": "image", "image": img} for img in cur_imgs] +
                  [txt(question_en)])
    msgs.append({"role": "user", "content": user_parts})

    return msgs, images

def safe_json(text: str):
    try:
        # Remove triple backtick wrapping
        if text.startswith("```"):
            text = text.strip("`").strip()

        # Extract JSON starting from first curly brace
        first_curly = text.index("{")
        text = text[first_curly:]

        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print(f"[WARN] Failed to parse JSON: {e}")
    return {"metadata": {}, "responses": [{"en": ""}]}




def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def main():
    print("[+] Loading Llama‑4 model and retriever…")
    metadata_dict = parse_metadata_dict(DICT_FILE)
    system_prompt = make_system_prompt(metadata_dict)
    retriever = Retriever()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=HF_TOKEN)
    model = Llama4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
    ).eval()

    test_cases = load_json(TEST_FILE)
    results = []
    

    ensure_dir(OUTPUT_FILE)
    ensure_dir(RAW_OUTPUT_FILE)

    with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as logf:
        for case in tqdm(test_cases, desc="cases"):
            encounter_id = case.get("encounter_id", "UNKNOWN")

            # === Check for at least one image ===
            img_paths = [os.path.join(IMAGES_ROOT, fn) for fn in case.get("image_ids", [])]
            if not any(os.path.exists(p) for p in img_paths):
                print(f"[SKIP] No images found for case {encounter_id}")
                continue

            # === Retrieve exemplars ===
            try:
                with Image.open(img_paths[0]) as img:
                    img_resized = img.convert("RGB").resize(TARGET_SIZE)
                    exemplars = retriever.retrieve_by_image(img_resized, top_k=2)
                    # query_text = case.get("query_title_en", "") + " " + case.get("query_content_en", "")
                    # exemplars = retriever.retrieve_hybrid(query_text, img_resized, top_k=2)

            except Exception as e:
                print(f"[ERROR] Retrieval failed for {encounter_id}: {e}")
                exemplars = []

            # === Construct prompt and image list ===
            try:
                messages, images = build_chat_and_images(system_prompt, exemplars, case, IMAGES_ROOT)
                chat_text = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = processor(text=chat_text, images=images, return_tensors="pt", padding=True).to(model.device)
            except Exception as e:
                print(f"[ERROR] Failed building prompt for {encounter_id}: {e}")
                continue

            # === Run model inference ===
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4092,
                    temperature=0.2,
                    top_p=0.9,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
                gen_text = processor.decode(outputs[0], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"[ERROR] Generation failed for {encounter_id}: {e}")
                gen_text = ""

            # === Parse model output ===
            parsed = safe_json(gen_text)

            # Normalize and safeguard parsed structure
            resp = parsed.get("responses")

            if isinstance(resp, str):
                parsed["responses"] = [{"en": resp}]
            elif isinstance(resp, dict):
                parsed["responses"] = [resp if "en" in resp else {"en": ""}]
            elif isinstance(resp, list):
                parsed["responses"] = [
                    r if isinstance(r, dict) and "en" in r else {"en": str(r) if r else ""}
                    for r in resp
                ]
            else:
                parsed["responses"] = [{"en": ""}]

            # Ensure metadata is valid
            if not isinstance(parsed.get("metadata"), dict):
                parsed["metadata"] = {}

            # Final fallback in case first response still lacks 'en'
            if "en" not in parsed["responses"][0]:
                parsed["responses"][0]["en"] = ""

            # === Package final result ===
            enriched = dict(case)
            enriched["split"] = "valid"
            enriched.update(parsed["metadata"])
            enriched["responses"] = [{
                "author_id": "assistant",
                "content_en": parsed["responses"][0]["en"]
            }]
            results.append(enriched)

            # === Format: write raw generation or fallback structure ===
            logf.write(f"encounter_id: {encounter_id}\n")

            if gen_text:
                logf.write(gen_text.strip() + "\n\n")
            else:
                empty_output = {
                    "anatomic_locations": [],
                    "wound_type": [],
                    "wound_thickness": [],
                    "tissue_color": [],
                    "drainage_amount": [],
                    "drainage_type": [],
                    "infection": [],
                    "responses": [{"en": ""}]
                }
                logf.write(json.dumps(empty_output, ensure_ascii=False, indent=2) + "\n\n")

    # === Save final structured outputs ===
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved {len(results)} predictions → {OUTPUT_FILE}")

    


if __name__ == "__main__":
    main()
