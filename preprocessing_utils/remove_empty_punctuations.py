import re

# === REGEX: EMPTY STRUCTURES PATTERN ===
# Matches any of the following when they contain ONLY whitespace (or nothing):
# - Empty parentheses: ()
# - Empty single quotes: ''
# - Empty double quotes: ""
# All of the above may contain optional spaces/tabs inside.
EMPTY_STRUCTURES_PATTERN = re.compile(
    r"\(\s*\)"       # Matches (   )
    r"|'(\s*)'"      # Matches '   '
    r'|\"(\s*)\"'    # Matches "   "
)

def remove_empty_structures(line: str) -> str:
    """
    Remove all occurrences of empty structures such as (), '', or ""
    that contain only whitespace (or are completely empty).

    Args:
        line (str): A line from the dataset.

    Returns:
        str: The line with empty structures removed.
    """
    return EMPTY_STRUCTURES_PATTERN.sub("", line)

def clean_file(filepath: str):
    """
    Clean an entire file by removing empty (), '', "" from each line.
    Overwrites the original file.

    Args:
        filepath (str): Path to the text file to process.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = [remove_empty_structures(line) for line in lines]

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Removed empty whitespace-only (), '', \"\" from: {filepath}")

# === MAIN ENTRY POINT ===
# Replace with your actual file path
if __name__ == "__main__":
    file_path = "/mnt/ml/SinBERT/output copy.txt"
    clean_file(file_path)
