import unicodedata
import re

# === REGEX: ALLOWED CHARACTERS DEFINITION ===
# This pattern removes any character that is NOT:
# - Sinhala Unicode block: U+0D80 to U+0DFF
# - ZWJ (U+200D) and ZWNJ (U+200C) — necessary for ligatures
# - Sinhala digits: ෦ to ෯
# - ASCII digits: 0 to 9
# - Common punctuation: . , ? ! : ; " ' ‘ ’ “ ” ( ) [ ] { } - … and whitespace

ALLOWED_PATTERN = re.compile(r"[^\u0D80-\u0DFF\u200C\u200D0-9෦-෯.,?!:;\"'‘’“”()\[\]{}\-…\s]")

def clean_sinhala_line(line: str) -> str:
    """
    Clean a single line by:
    1. Applying Unicode NFC normalization to merge characters correctly.
    2. Removing all characters not defined in the allowed set above.
    
    Args:
        line (str): A raw Sinhala text line.

    Returns:
        str: Cleaned version with only allowed Sinhala and punctuation.
    """
    line = unicodedata.normalize("NFC", line)  # Normalize to canonical form
    return ALLOWED_PATTERN.sub("", line)       # Remove disallowed characters

def clean_file_in_place(filepath: str):
    """
    Clean all lines in a text file in-place (overwrite original).
    
    Args:
        filepath (str): Path to the text file to be cleaned.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = [clean_sinhala_line(line) for line in lines]

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

# === MAIN EXECUTION ===
# Replace with your actual file path
if __name__ == "__main__":
    file_path = "/mnt/ml/SinBERT/output copy.txt"
    clean_file_in_place(file_path)
    print(f"Cleaned: {file_path}")
