---
base_model: facebook/bart-large-cnn
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:facebook/bart-large-cnn
- lora
- transformers
---
# Research Paper Summarizer AI Agent

An intelligent multi-agent system that automates research paper analysis, summarization, and bibliography management using fine-tuned language models.

## Project Overview

This AI agent system addresses the daily challenge of reading and summarizing academic papers by:
- Automatically parsing PDF research papers
- Generating structured summaries using domain-adapted models
- Extracting and formatting citations
- Managing bibliography in multiple formats
- Providing semantic search across paper collections

## Architecture

### Multi-Agent System
- **Planner Agent**: Analyzes paper structure and creates processing strategy
- **Extractor Agent**: Handles PDF parsing and content extraction
- **Summarizer Agent**: Fine-tuned model for academic summarization
- **Bibliography Agent**: Citation extraction and formatting

### Key Features
- **Fine-tuned Model**: LoRA-adapted language model specialized for academic content
- **RAG Integration**: Vector database for contextual paper retrieval
- **Evaluation Metrics**: ROUGE scores, citation accuracy, domain relevance
- **Web Interface**: User-friendly dashboard for paper upload and results

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### CLI Interface
```bash
# Process a single paper
python main.py --input paper.pdf --output summary.json

# Batch process multiple papers
python main.py --batch papers/ --output results/
```

### Web Interface
```bash
# Start the web server
streamlit run app.py
```

## Project Structure

```
research-paper-summarization/
├── src/
│   ├── agents/          # Multi-agent system
│   ├── models/          # Fine-tuning and inference
│   ├── parsers/         # PDF processing
│   ├── evaluation/      # Metrics and testing
│   └── utils/           # Helper functions
├── data/
│   ├── raw/             # Original papers
│   ├── processed/       # Extracted content
│   └── training/        # Fine-tuning datasets
├── models/              # Saved model checkpoints
├── tests/               # Unit and integration tests
└── notebooks/           # Jupyter notebooks for analysis
```

## Fine-tuning Details

The project uses LoRA (Low-Rank Adaptation) to fine-tune a base language model on academic content:
- **Base Model**: Llama-2-7B or Mistral-7B
- **Dataset**: ArXiv papers and academic abstracts
- **Objective**: Improve domain-specific summarization quality
- **Rationale**: Academic writing has unique terminology and structure patterns

## Evaluation

The system is evaluated using multiple metrics:
- **ROUGE Scores**: Content overlap with reference summaries
- **Citation Accuracy**: Precision/recall of extracted references
- **Domain Relevance**: Semantic similarity using SciBERT embeddings
- **User Studies**: Quality ratings from academic users

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
### Framework versions

- PEFT 0.17.1