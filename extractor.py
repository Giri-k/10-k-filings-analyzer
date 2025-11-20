from bs4 import BeautifulSoup
import re
import os

# Item 7 – Management’s Discussion & Analysis
# Item 1A – Risk Factors
# Item 1 – Business
# Item 7A – Quantitative and Qualitative Disclosures About Market Risk
# Item 8 – Financial Statements and Supplementary Data
# Item 9A – Controls and Procedures
important_items = [
    "ITEM 1A", "ITEM 7", "ITEM 7A", "ITEM 8", "ITEM 1", "ITEM 9A"
]

def load_and_clean(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    #print(f"Text: {text}")
    return text

def split_10k_sections(text):
    # flexible pattern: ITEM 1, ITEM 1A., ITEM 7A etc.
    pattern = r"(ITEM\s+[0-9A-Z]+\.*\s+[A-Z][A-Z\s,&\-]+)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = {}
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        sections[heading] = body
    return sections

def filter_sections(sections):
    filtered = {}
    for key, val in sections.items():
        for imp in important_items:
            if re.search(imp, key, re.IGNORECASE):
                filtered[imp] = val
    return filtered

def save_sections(filtered, company="AAPL", year="2024"):
    os.makedirs("sections", exist_ok=True)
    for key, content in filtered.items():
        fname = key.replace(" ", "_").replace(".", "")  # e.g. ITEM_7A
        path = f"sections/{company}_{year}_{fname}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

def extract_year_from_folder(folder_name):
    """Extract 2-digit year from folder like 0000320193-20-000096 → returns 2020"""
    match = re.search(r"-(\d{2})-", folder_name)
    if match:
        yr = int(match.group(1))
        return f"20{yr:02d}"  # convert 20 → 2020, 21 → 2021, etc.
    return "unknown"

def process_10k_filings(company):
    dir = f"./sec-edgar-filings/{company}/10-K"

    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist")
        return
    
    for sub_dir in os.listdir(dir):
        sub_path = os.path.join(dir, sub_dir)
        if os.path.isdir(sub_path):
            year = extract_year_from_folder(sub_dir)
            file_path = os.path.join(sub_path, "full-submission.txt")
            print(f"Processing {company} {year} {file_path}")
            if os.path.exists(file_path):
                text = load_and_clean(file_path)
                sections = split_10k_sections(text)
                filtered = filter_sections(sections)
                save_sections(filtered, company=company, year=year)
            else:
                print(f"File {file_path} does not exist")
    print(f"Processed {company} 10-K filings and saved to sections folder")

if __name__ == '__main__':
    #./sec-edgar-filings/AAPL/10-K/0000320193-20-000096/full-submission.txt
    process_10k_filings("AAPL")