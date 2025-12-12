import re

def remove_extra_spaces(line: str) -> str:
    """
    Replaces multiple spaces, tabs, and newlines in a line with a single space.
    Also trims leading and trailing whitespace.

    Args:
        line (str): Raw input line from text file.

    Returns:
        str: Cleaned line with normalized spacing.
    """
    return re.sub(r"\s+", " ", line).strip()

def clean_spaces_in_file(filepath: str):
    """
    Reads a text file, removes unnecessary extra whitespace in each line,
    and writes the cleaned output back to the same file.

    Args:
        filepath (str): Path to the text file to clean.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Apply whitespace cleanup for each line
    cleaned_lines = [remove_extra_spaces(line) for line in lines]

    # Ensure each line ends with a newline character
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines([line + "\n" for line in cleaned_lines])

    print(f"Removed extra spaces from: {filepath}")

# === MAIN EXECUTION ===
# Replace with your actual dataset file
if __name__ == "__main__":
    file_path = "/mnt/ml/SinBERT/output copy.txt"
    clean_spaces_in_file(file_path)
