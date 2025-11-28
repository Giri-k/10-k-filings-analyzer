# 10-K Filings Analyzer

A Retrieval-Augmented Generation (RAG) application for analyzing SEC 10-K filings using open-source models. This tool extracts key financial information from annual reports and provides intelligent insights through semantic search and AI-powered analysis.

## Overview

This project implements a RAG (Retrieval-Augmented Generation) pipeline to analyze SEC 10-K filings. It downloads filings, extracts important sections, creates vector embeddings, and uses a language model to generate comprehensive financial insights based on user queries.

## Features

- **Automated 10-K Download**: Fetches up to 5 years of 10-K filings from SEC EDGAR
- **Intelligent Section Extraction**: Automatically extracts key sections (Items 1, 1A, 7, 7A, 8, 9A) from filings
- **Vector Search**: Uses ChromaDB for efficient semantic search across extracted content
- **Open-Source Models**: Leverages Hugging Face models (FLAN-T5) for generation and Sentence Transformers for embeddings
- **Interactive Interface**: Gradio-based web interface for easy querying
- **GPU Support**: Automatically uses CUDA if available for faster processing

## Architecture

The application follows a RAG pipeline:

1. **Download**: Fetches 10-K filings from SEC EDGAR database
2. **Extract**: Parses HTML and extracts important sections using regex patterns
3. **Chunk**: Splits documents into overlapping chunks for better retrieval
4. **Embed**: Creates vector embeddings using Sentence Transformers
5. **Store**: Persists embeddings in ChromaDB vector database
6. **Retrieve**: Performs semantic search to find relevant context
7. **Generate**: Uses FLAN-T5 model to generate insights based on retrieved context

## Tech Stack

### Core Libraries
- **sec-edgar-downloader**: Downloads SEC filings
- **BeautifulSoup**: HTML parsing and text extraction
- **LangChain**: Text splitting utilities
- **ChromaDB**: Vector database for embeddings

### AI/ML Components
- **Sentence Transformers**: `all-MiniLM-L6-v2` for embeddings
- **Hugging Face Transformers**: `google/flan-t5-large` for text generation
- **PyTorch**: Deep learning framework

### Frontend
- **Gradio**: Web interface for user interaction

## Project Structure

```
10-k-filings-analyzer/
├── app.py              # Gradio frontend application
├── agent.py            # RAG agent with retrieval and generation logic
├── downloader.py       # Downloads 10-K filings from SEC
├── extractor.py        # Extracts key sections from 10-K filings
├── vectordb.py         # Builds and manages ChromaDB vector database
├── sec-edgar-filings/  # Downloaded filings (generated)
├── sections/           # Extracted text sections (generated)
└── chroma_db/          # Vector database storage (generated)
```

## Key Sections Analyzed

The application focuses on six critical sections from 10-K filings:

- **Item 1**: Business Overview
- **Item 1A**: Risk Factors
- **Item 7**: Management's Discussion and Analysis
- **Item 7A**: Quantitative and Qualitative Disclosures About Market Risk
- **Item 8**: Financial Statements and Supplementary Data
- **Item 9A**: Controls and Procedures

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)
- Internet connection (for downloading filings and models)

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd 10-k-filings-analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install sec-edgar-downloader beautifulsoup4 chromadb sentence-transformers transformers torch gradio langchain tqdm
   ```

   Or create a `requirements.txt` with:
   ```
   sec-edgar-downloader
   beautifulsoup4
   chromadb
   sentence-transformers
   transformers
   torch
   gradio
   langchain
   tqdm
   ```

3. **Install PyTorch** (if using GPU):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Running the Application

1. **Start the Gradio interface**:
   ```bash
   python app.py
   ```

2. **Use the web interface**:
   - Enter a company ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`)
   - Enter your question about the company's 10-K filings
   - Click "Submit" and wait for the analysis
   - First run will download and process filings automatically

### Example Queries

- "What are Apple's major risk factors mentioned?"
- "What are the key financial trends in Microsoft's recent filings?"
- "What risks does the company identify in Item 1A?"
- "What does management discuss about future outlook?"

### Command Line Usage

You can also use the agent directly:

```python
from agent import call_agent

insights = call_agent("AAPL", "What are Apple's major risk factors?")
print(insights)
```

## How It Works

### 1. Download Phase
- Uses `sec-edgar-downloader` to fetch the most recent 5 years of 10-K filings
- Files are stored in `sec-edgar-filings/{SYMBOL}/10-K/`

### 2. Extraction Phase
- Parses HTML from `full-submission.txt` files
- Uses regex patterns to identify and extract key sections
- Saves extracted sections to `sections/` directory

### 3. Vector Database Phase
- Loads extracted text files
- Splits documents into chunks (1000 characters with 200 overlap)
- Generates embeddings using Sentence Transformers
- Stores in ChromaDB for fast retrieval

### 4. Query Phase
- Encodes user query into embedding space
- Retrieves top-k most relevant chunks (default: 5)
- Constructs prompt with context and question
- Generates response using FLAN-T5 model

## Configuration

### Model Settings

In `agent.py`:
- `MODEL_NAME`: Change the Hugging Face model (default: `"google/flan-t5-large"`)
- `DEVICE`: Automatically detects CUDA availability
- `max_len`: Maximum tokens in generated response (default: 512)

### Vector Database Settings

In `vectordb.py`:
- `CHUNK_SIZE`: Characters per chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBED_MODEL`: Embedding model (default: `"all-MiniLM-L6-v2"`)
- `COLLECTION_NAME`: ChromaDB collection name

### Download Settings

In `downloader.py`:
- `limit`: Number of years to download (default: 5)

## Performance Considerations

- **First Run**: Downloads filings and models, which may take several minutes
- **Subsequent Runs**: Uses cached models and filings for faster response
- **GPU Acceleration**: Significantly speeds up embedding and generation
- **Memory**: FLAN-T5-Large requires ~3GB RAM; GPU memory recommended

## Limitations

- Processes up to 5 years of filings (configurable)
- Focuses on specific sections due to token limitations
- Model responses may vary in quality depending on query complexity
- Processing time depends on filing size and hardware

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_len` in `generate_insights()`
- Use a smaller model (e.g., `google/flan-t5-base`)
- Process fewer filings

### Slow Performance
- Ensure GPU is being used (check `DEVICE` variable)
- Reduce `top_k` retrieval count
- Use a smaller embedding model

### Missing Sections
- Some filings may have different formatting
- Check `extractor.py` regex patterns
- Verify `full-submission.txt` files exist

## Future Enhancements

- Support for additional filing types (10-Q, 8-K)
- Multi-company comparison queries
- Export analysis to PDF/CSV
- Fine-tuned models for financial analysis
- Support for more embedding models

## License

This project is for educational and research purposes.

## Acknowledgments

- SEC EDGAR database for public access to filings
- Hugging Face for open-source models and transformers library
- ChromaDB team for the vector database
- Sentence Transformers community

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
