import os
from sec_edgar_downloader import Downloader

EDGAR_COMPANY = os.environ.get("EDGAR_COMPANY", "University of Texas at Austin")
EDGAR_EMAIL = os.environ.get("EDGAR_EMAIL", "girishk1999@utexas.edu")

dl = Downloader(EDGAR_COMPANY, EDGAR_EMAIL)


def download(symbol: str):
    print(f"Downloading 10-K filings for {symbol}...", flush=True)
    dl.get("10-K", symbol, limit=5)
    print(f"Successfully downloaded 10-K filings for {symbol}")


if __name__ == "__main__":
    symbol = input("Enter the symbol: ")
    download(symbol)
