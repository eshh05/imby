# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- 10GB+ free disk space for models

## Step-by-Step Setup

### 1. Create Virtual Environment
```bash
# Navigate to project directory
cd /Users/amaverma/learn/ai/research-paper-summarization

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify activation (should show venv in prompt)
which python
```

### 2. Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will install:
# - PyTorch and Transformers for ML models
# - Streamlit for web interface
# - PDF processing libraries
# - Evaluation metrics libraries
# - Vector database components
```

### 3. Download Required NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 4. Test Installation
```bash
# Run quick test to verify everything works
python quick_test.py
```

## Quick Start (After Setup)

### Web Interface
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Launch web interface
streamlit run app.py
```

### Command Line Interface
```bash
# Process a single paper
python main.py process sample_paper.pdf

# Launch web interface via CLI
python main.py web
```

### 2. Download Required Models
The system will automatically download models on first use, but you can pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 3. Optional: Fine-tune Your Own Model
```bash
# Create sample training data and fine-tune
python main.py fine-tune --epochs 3 --batch-size 2
```

### 4. Optional: Setup Vector Store
```bash
# Initialize vector store for RAG
python main.py vector-store --output data/vector_store
```

## Usage Examples

### Web Interface
```bash
streamlit run app.py
```
- Upload PDF papers
- Get structured summaries
- View citations and bibliography
- Search similar papers

### Command Line Interface

**Process Single Paper:**
```bash
python main.py process paper.pdf --output summary.json
```

**Batch Processing:**
```bash
python main.py batch papers_folder/ --output results/
```

**Run Evaluation:**
```bash
python main.py evaluate --test-data data/test
```

**Fine-tune Model:**
```bash
python main.py fine-tune --train-data training.json --epochs 5
```

## Configuration

### Model Configuration
Edit the model settings in your code:
```python
config = {
    'summarizer': {
        'model_name': 'microsoft/DialoGPT-medium',  # Change base model
        'max_length': 512
    },
    'extractor': {
        'method': 'pdfplumber'  # or 'pymupdf', 'pypdf2'
    }
}
```

### RAG Configuration
```python
vector_store_config = {
    'collection_name': 'research_papers',
    'persist_directory': 'data/vector_store',
    'embedding_model': 'all-MiniLM-L6-v2'
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in fine-tuning
   - Use CPU-only mode: set `device_map="cpu"`

2. **PDF Extraction Fails**
   - Try different extraction methods: `--extraction-method pymupdf`
   - Ensure PDF is not corrupted or password-protected

3. **Model Loading Issues**
   - Check internet connection for model downloads
   - Verify sufficient disk space (models can be 1-7GB)

4. **Dependencies Issues**
   ```bash
   pip install --upgrade transformers torch
   ```

### Performance Tips

1. **For Better Speed:**
   - Use GPU if available
   - Reduce model size for faster inference
   - Enable model quantization

2. **For Better Quality:**
   - Fine-tune on domain-specific data
   - Use larger base models
   - Increase context window size

## System Requirements

- **Python:** 3.8+
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 10GB for models and data
- **GPU:** Optional but recommended for fine-tuning

## Project Structure

```
research-paper-summarization/
├── src/
│   ├── agents/          # Multi-agent system
│   ├── models/          # Fine-tuning pipeline
│   ├── rag/             # Vector store and retrieval
│   └── evaluation/      # Metrics and testing
├── data/                # Training and test data
├── models/              # Saved model checkpoints
├── results/             # Output results
├── app.py              # Streamlit web interface
├── main.py             # CLI interface
└── requirements.txt    # Dependencies
```
