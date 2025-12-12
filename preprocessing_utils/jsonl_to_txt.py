import json

def jsonl_to_txt(jsonl_path, txt_path):
    """
    in this code we read a jsonl file line by line, extract the "text" field from each JSON object,
    remove the first two lines from the text, and write the cleaned text to a new txt file.
    """
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file, \
         open(txt_path, "w", encoding="utf-8") as txt_file:

        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                text = obj.get("text", "")

                # Split into lines
                parts = text.split("\\n")

                # Remove first two lines
                cleaned = "\\n".join(parts[2:])

                txt_file.write(cleaned + "\n")

            except json.JSONDecodeError:
                print("Skipping invalid JSON:", line)

if __name__ == "__main__":
    jsonl_to_txt("/mnt/ml/SinBERT/si_clean_0000.jsonl", "output.txt")
