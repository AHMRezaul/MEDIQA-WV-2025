import json
import re

# === File paths ===
input_json_path = "dataset/mediqa2025-wv-testinputs/test_inputonly.json"
raw_output_path = "output.txt"
output_json_path = "prediction.json"

# === Load test input data ===
with open(input_json_path, "r", encoding="utf-8") as f:
    input_data = {entry["encounter_id"]: entry for entry in json.load(f)}

# === Load raw generations and split into encounter blocks ===
with open(raw_output_path, "r", encoding="utf-8") as f:
    content = f.read()

# Split on 'encounter_id: ...' lines
blocks = re.split(r"encounter_id:\s*(ENC\d+)", content)[1:]  # skip initial empty
structured_outputs = []

# Process blocks in pairs: [encounter_id, block_content]
for i in range(0, len(blocks), 2):
    encounter_id = blocks[i].strip()
    block = blocks[i + 1]

    # Try to extract JSON block
    json_match = re.search(r"\{.*\}", block, re.DOTALL)
    if not json_match:
        print(f"[WARN] No JSON found for {encounter_id}, using empty output.")
        parsed = {
            "metadata": {
                "anatomic_locations": [],
                "wound_type": [],
                "wound_thickness": [],
                "tissue_color": [],
                "drainage_amount": [],
                "drainage_type": [],
                "infection": []
            },
            "responses": [""]
        }
    else:
        try:
            parsed = json.loads(json_match.group())
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON for {encounter_id}: {e}")
            continue

    # Safely extract metadata and response
    md = parsed.get("metadata", {})
    resp = parsed.get("responses", [""])
    if isinstance(resp, list):
        content_en = resp[0] if resp else ""
    else:
        content_en = resp

    # Retrieve matching input entry
    case = input_data.get(encounter_id)
    if not case:
        print(f"[ERROR] Encounter ID {encounter_id} not found in input file.")
        continue

    # Assemble structured output
    formatted = {
        "encounter_id": encounter_id,
        "split": "valid",
        "image_ids": case.get("image_ids", []),
        "anatomic_locations": md.get("anatomic_locations", []),
        "wound_type": md.get("wound_type", [""])[0] if md.get("wound_type") else "",
        "wound_thickness": md.get("wound_thickness", [""])[0] if md.get("wound_thickness") else "",
        "tissue_color": md.get("tissue_color", [""])[0] if md.get("tissue_color") else "",
        "drainage_amount": md.get("drainage_amount", [""])[0] if md.get("drainage_amount") else "",
        "drainage_type": md.get("drainage_type", [""])[0] if md.get("drainage_type") else "",
        "infection": md.get("infection", [""])[0] if md.get("infection") else "",
        "query_title_en": case.get("query_title_en", ""),
        "query_title_zh": case.get("query_title_zh", ""),
        "query_content_en": case.get("query_content_en", ""),
        "query_content_zh": case.get("query_content_zh", ""),
        "responses": [
            {
                "author_id": "assistant",
                "content_en": content_en.strip(),
                "content_zh": ""
            }
        ]
    }

    structured_outputs.append(formatted)

# === Save to output file ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(structured_outputs, f, ensure_ascii=False, indent=2)

print(f"[✓] Saved {len(structured_outputs)} formatted predictions → {output_json_path}")
