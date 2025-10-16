from sec_edgar_downloader import Downloader

dl = Downloader("University of Texas at Austin", "girishk1999@utexas.edu")

def download(symbol: str):
    print(f"Downloading 10-K filings for {symbol}…", flush=True)
    dl.get("10-K", symbol, limit=5)
    print(f"Successfully downloaded 10-K filings for {symbol}")

if __name__ == '__main__':
    # take input from command line
    symbol = input("Enter the symbol: ")
    download(symbol)