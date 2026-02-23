import os
import re
import unicodedata
import concurrent.futures

# === CONFIGURATION ===

# Path to your input dataset (should be a plain .txt file with one sentence or paragraph per line)
INPUT_FILE = "/mnt/ml/HelaBERT/output copy.txt"

# Output file for the normalized text
# IMPORTANT: Keep the ".normalized.txt" suffix to indicate the file is processed
OUTPUT_FILE = "output_path.normalized.txt"

# Use all available CPU cores for parallel normalization
NUM_THREADS = os.cpu_count()

# === REGEX DEFINITIONS ===

# Regex to remove unwanted invisible Unicode characters except ZWJ (U+200D),
# which is important for proper Sinhala script ligature rendering
REMOVE_ZW_CHARS_RE = re.compile(r"[\u200B\u200C\u200E\u200F\uFEFF]")

# === NORMALIZATION FUNCTION ===

def normalize_line(line: str) -> str:
    """
    Normalize a single line of text by:
    1. Removing zero-width and directional invisible characters (except ZWJ).
    2. Applying Unicode NFC normalization to combine characters properly.
    
    Args:
        line (str): A raw input string from the dataset.

    Returns:
        str: Cleaned and normalized version of the input string.
    """
    line = REMOVE_ZW_CHARS_RE.sub("", line)  # Remove unwanted characters
    return unicodedata.normalize("NFC", line)  # Convert to NFC form

# === FILE NORMALIZATION FUNCTION ===

def normalize_file(input_path: str, output_path: str, threads: int = 4):
    """
    Normalize an entire text file using multithreading.
    
    Args:
        input_path (str): Path to the raw input .txt file.
        output_path (str): Path to write the normalized output.
        threads (int): Number of threads to use for parallel processing.
    """
    print(f"Reading input file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    print(f"Normalizing with {threads} threads...")
    # Normalize lines in parallel using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        normalized_lines = list(executor.map(normalize_line, lines))

    print(f"Writing normalized output to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        # Ensure all lines end with a newline character
        fout.writelines(line if line.endswith("\n") else line + "\n" for line in normalized_lines)

    print("Done! Text normalization completed.")

# === MAIN ENTRY POINT ===

if __name__ == "__main__":
    # Run the normalization process using all CPU cores
    normalize_file(INPUT_FILE, OUTPUT_FILE, threads=NUM_THREADS)
