import re

def clean_repeated_punctuation(file_path: str):
    """
    Cleans a text file by reducing repeated punctuation (with optional spaces)
    into a single punctuation mark followed by a space.

    Example:
        "What???"         → "What? "
        "Wow!!  !!"       → "Wow! "
        "Hello... World"  → "Hello. World"

    Args:
        file_path (str): Path to the text file to clean.
    """

    # Define all standard punctuation characters (escaped)
    punctuations = r"[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]"

    # This pattern matches 2 or more repeated punctuations, allowing spaces between them
    pattern = re.compile(rf"(({punctuations})\s*){{2,}}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []

    for line in lines:
        # Replace repeated punctuation blocks with a single character + space
        def replacer(match):
            punct = match.group(2)  # the actual punctuation character
            return punct + " "      # replace with one and a space

        cleaned_line = pattern.sub(replacer, line)
        cleaned_lines.append(cleaned_line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"Repeated punctuations cleaned in: {file_path}")

# === MAIN EXECUTION ===
# Replace with your actual file path
if __name__ == "__main__":
    clean_repeated_punctuation("/mnt/ml/HelaBERT/output copy.txt")