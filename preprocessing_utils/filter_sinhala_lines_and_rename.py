import re
import os

def has_sinhala(text: str) -> bool:
    """
    Check if the given text contains at least one Sinhala Unicode character.

    Sinhala block range: U+0D80–U+0DFF
    """
    return re.search(r"[\u0D80-\u0DFF]", text) is not None

def is_valid_line(line: str) -> bool:
    """
    Determine if a line is valid:
    - Non-empty after stripping.
    - At least 5 characters long.
    - Contains Sinhala characters.

    Args:
        line (str): A single line from the file.

    Returns:
        bool: True if the line is valid and should be kept.
    """
    stripped = line.strip()
    return bool(stripped) and len(stripped) >= 5 and has_sinhala(stripped)

def get_word_count(text: str) -> int:
    """
    Calculate word count from a cleaned string.

    Args:
        text (str): Multiline cleaned text.

    Returns:
        int: Total number of words.
    """
    return len(text.split())

def clean_and_rename_file(filepath: str):
    """
    Clean the input file by:
    - Keeping only valid Sinhala lines.
    - Stripping extra whitespace.
    - Saving the cleaned result to a new file with the format: `<line_count>_<word_count>.txt`

    Args:
        filepath (str): Path to the raw input file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter valid lines (with Sinhala text, min length 5)
    cleaned_lines = [line.strip() for line in lines if is_valid_line(line)]

    # Join lines and compute stats
    final_text = "\n".join(cleaned_lines)
    line_count = len(cleaned_lines)
    word_count = get_word_count(final_text)

    # Construct new filename based on counts
    dir_path = os.path.dirname(filepath)
    new_filename = f"{line_count}_{word_count}.txt"
    new_path = os.path.join(dir_path, new_filename)

    # Save the cleaned content
    with open(new_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"Saved cleaned file as: {new_filename}")

# === MAIN EXECUTION ===
# Replace with the path to your actual input file
if __name__ == "__main__":
    file_path = "add_file_path_(file_should_be_in_.txt_format)"
    clean_and_rename_file(file_path)