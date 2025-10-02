# remove_prompt_block.py

def remove_instruction_block(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    # Define the exact start of the instruction block
    start_phrase = "system"
    end_phrase = "Do not add new fields or invent new labels. Only use allowed metadata values."

    inside_block = False
    cleaned_lines = []

    for line in lines:
        if line.strip() == start_phrase:
            inside_block = True
            continue
        if inside_block and end_phrase in line:
            inside_block = False
            continue
        if not inside_block:
            cleaned_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(cleaned_lines)

# Example usage:
remove_instruction_block("raw_generations.txt", "output.txt")
