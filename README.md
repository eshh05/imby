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
