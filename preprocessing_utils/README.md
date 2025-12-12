# Sinhala Dataset Cleaning Utilities

A collection of lightweight Python scripts to preprocess and clean Sinhala text data  
before tokenizer training and language model development.

---

## Included Scripts

- **clean_repeated_punctuation.py**  
  Normalize repeated punctuation marks (e.g., "!!!" → "! ")

- **filter_sinhala_lines_and_rename.py**  
  Keep only lines containing Sinhala characters with minimum length; saves cleaned files with line and word counts in the filename.

- **remove_empty_punctuations.py**  
  Remove empty parentheses `()`, single quotes `''`, and double quotes `""` including whitespace-only cases.

- **remove_extra_whitespace.py**  
  Collapse multiple spaces, tabs, and newlines into a single space; trims leading/trailing spaces.

- **remove_non_sinhala_characters.py**  
  Unicode NFC normalization + remove all characters except Sinhala script, Sinhala digits, select punctuation, and digits.

- **remove_unmatched_brackets_and_dates.py**  
  Remove date-like patterns `(2023)` and unmatched brackets or quotes while preserving valid pairs.

- **split_lines_by_length.py**  
  Categorize lines into short (<5 words), main (5–50 words), and long (>50 words) and save them into separate files.

- **text_normalizer.py**  
  Remove zero-width and invisible Unicode chars except ZWJ; normalize text using Unicode NFC with multithreading.

---

## Usage

Each script is standalone.  
Edit the `file_path` or input/output variables inside each script, then run:

```
python script_name.py
```

## Notes
- All scripts overwrite files or save new cleaned versions with descriptive filenames.
- Designed specifically for Sinhala datasets but can be adapted.
- Use these tools sequentially to maximize data quality before tokenizer or model training.