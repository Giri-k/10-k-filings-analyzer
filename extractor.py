from bs4 import BeautifulSoup
import re
import os

CLEANED_DIR = "./cleaned_filings"


def load_and_clean(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_year_from_folder(folder_name):
    match = re.search(r"-(\d{2})-", folder_name)
    if match:
        yr = int(match.group(1))
        return f"20{yr:02d}"
    return "unknown"


def process_10k_filings(company):
    base_dir = f"./sec-edgar-filings/{company}/10-K"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} does not exist")

    os.makedirs(CLEANED_DIR, exist_ok=True)

    for sub_dir in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        year = extract_year_from_folder(sub_dir)
        file_path = os.path.join(sub_path, "full-submission.txt")
        if os.path.exists(file_path):
            print(f"Processing {company} {year}")
            text = load_and_clean(file_path)
            out_path = os.path.join(CLEANED_DIR, f"{company}_{year}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(f"File {file_path} does not exist")

    print(f"Cleaned {company} 10-K filings → {CLEANED_DIR}/")


if __name__ == "__main__":
    process_10k_filings("AAPL")
