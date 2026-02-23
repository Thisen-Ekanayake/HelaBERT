import re

# === REGEX: PATTERNS TO REMOVE ===

# Matches numeric-like patterns inside parentheses — e.g., (2023), ( 99 ), etc.
DATE_PATTERN = re.compile(r"\(\s*\d{1,4}\s*\)")

def remove_unmatched_pairs(line: str) -> str:
    """
    Cleans a line by:
    1. Removing year/date-like patterns in the form of (2023), ( 1999 ), etc.
    2. Removing unmatched brackets and quotes while preserving valid ones.

    Args:
        line (str): A raw text line.

    Returns:
        str: Cleaned line with unmatched or noisy bracket/quote structures removed.
    """
    # Step 1: Remove date-like expressions (e.g., (2023))
    line = DATE_PATTERN.sub("", line)

    # Step 2: Remove unmatched brackets and quotes while preserving valid ones

    # Define patterns for valid pairs to preserve
    pairs = {
        '"': r'"[^"]+"',       # Double quotes: "something"
        "'": r"'[^']+'",       # Single quotes: 'something'
        r'\(': r'\([^()]+\)',  # Round brackets: (content)
        r'\[': r'\[[^\[\]]+\]',# Square brackets: [content]
        r'\{': r'\{[^{}]+\}',  # Curly brackets: {content}
    }

    preserved = []

    # Extract valid bracket/quote pairs and temporarily replace with placeholders
    for symbol, pattern in pairs.items():
        for match in re.findall(pattern, line):
            preserved.append(match)
            line = line.replace(match, '☯️', 1)  # Use unique placeholder to mark preserved content

    # Remove any remaining unmatched brackets or quotes
    line = re.sub(r"[(){}\[\]\"']", "", line)

    # Restore the preserved (valid) pairs back to their original positions
    for match in preserved:
        line = line.replace('☯️', match, 1)

    return line.strip()

def clean_file(filepath: str):
    """
    Cleans a text file by applying the `remove_unmatched_pairs` function
    to every line. Overwrites the file in-place.

    Args:
        filepath (str): Path to the file you want to clean.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = [remove_unmatched_pairs(line) + "\n" for line in lines]

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(cleaned)

    print(f"Fully cleaned: {filepath}")

# === MAIN ENTRY POINT ===
# Replace with your actual file path
if __name__ == "__main__":
    file_path = "/mnt/ml/HelaBERT/output copy.txt"
    clean_file(file_path)
