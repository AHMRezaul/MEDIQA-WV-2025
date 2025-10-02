import re
import json

def clean_and_format_json_array(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Step 1: Remove lines starting with `[`
    cleaned_lines = [
        line for line in lines
        if not line.lstrip().startswith("[")
    ]

    # Step 2: Join and replace unwanted patterns
    cleaned_text = ''.join(cleaned_lines)
    cleaned_text = cleaned_text.replace("Raw output:", "")
    cleaned_text = cleaned_text.replace("```", ",")

    # Step 3: Wrap in brackets to form a JSON array
    wrapped_text = "[\n" + cleaned_text.strip().rstrip(",") + "\n]"

    # Step 4: Save to output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(wrapped_text)

    print(f"[+] Cleaned and saved JSON array → {output_path}")


# Example usage:
if __name__ == "__main__":
    input_file = "output.out"      # Replace with your input file
    output_file = "prediction.json"   # Desired output path
    clean_and_format_json_array(input_file, output_file)
