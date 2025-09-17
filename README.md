# Eshita Verma
# 2023BB10808
# IIT Delhi
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

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)

An intelligent multi-agent system that automates research paper analysis, summarization, and bibliography management using fine-tuned language models.

---

## Overview

This AI agent system addresses the daily challenge of reading and summarizing academic papers by:

- Automatically parsing PDF research papers
- Generating structured summaries using domain-adapted models
- Extracting and formatting citations
- Managing bibliography in multiple formats
- Providing semantic search across paper collections

*For a detailed assignment report, see [ASSIGNMENT_FULFILLMENT.md](ASSIGNMENT_FULFILLMENT.md)*

---

## Architecture

### Multi-Agent System

- **Planner Agent**: Analyzes paper structure and creates processing strategy
- **Extractor Agent**: Handles PDF parsing and content extraction
- **Summarizer Agent**: Fine-tuned model for academic summarization
- **Bibliography Agent**: Citation extraction and formatting
- **Orchestrator**: Coordinates the workflow and error handling

**Workflow Diagram:**
PDF Upload → Extractor → Planner → Summarizer (+RAG Context) → Bibliography → Output

### Key Features

- **Fine-tuned Model**: LoRA-adapted language model specialized for academic content
- **RAG Integration**: Vector database for contextual paper retrieval
- **Evaluation Metrics**: ROUGE scores, citation accuracy, domain relevance
- **Web Interface**: User-friendly dashboard for paper upload and results

*For detailed engineering design, see [ENGINEERING_DESIGN.md](ENGINEERING_DESIGN.md)*

---

## Installation

### Step 1: Create Virtual Environment
cd /path/to/research-paper-summarization  
python -m venv venv  

 Activate virtual environment  
 Linux/macOS: source venv/bin/activate  
 Windows: venv\Scripts\activate  

### Step 2: Install Dependencies
pip install --upgrade pip  
pip install -r requirements.txt  

### Step 3: Download Required NLTK Data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"  

### Step 4: Test Installation
python quick_test.py  

*For complete setup instructions, see [SETUP.md](SETUP.md)*

---

## Usage

### CLI Interface
python main.py --input paper.pdf --output summary.json  
python main.py --batch papers/ --output results/  
python main.py evaluate --test-data data/test  

### Web Interface
streamlit run app.py  

- Upload PDF papers  
- Get structured summaries  
- View citations and bibliography  
- Semantic search powered by RAG  

---

## Project Structure

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

---

## Fine-tuning Details

The project uses LoRA (Low-Rank Adaptation) to fine-tune a base language model on academic content:

- **Base Model**: Llama-2-7B or Mistral-7B  
- **Dataset**: ArXiv papers and academic abstracts  
- **Objective**: Improve domain-specific summarization quality  
- **Rationale**: Academic writing has unique terminology and structure patterns  

---

## Evaluation

The system is evaluated using multiple metrics:

- **ROUGE Scores**: Content overlap with reference summaries  
- **Citation Accuracy**: Precision/recall of extracted references  
- **Domain Relevance**: Semantic similarity using SciBERT embeddings  
- **User Studies**: Quality ratings from academic users  

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Add tests for new functionality  
4. Run the test suite: `pytest tests/`  
5. Submit a pull request  

---


*For more setup and technical guidance, see [SETUP.md](SETUP.md)*  
*For detailed design, see [ENGINEERING_DESIGN.md](ENGINEERING_DESIGN.md)*  
*For assignment fulfillment details, see [ASSIGNMENT_FULFILLMENT.md](ASSIGNMENT_FULFILLMENT.md)*
