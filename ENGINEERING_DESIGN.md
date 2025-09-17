# Engineering Design Document
## Research Paper Summarization AI Agent

**Version:** 1.0  
**Date:** 2025-01-13  
**Author:** AI Agent Development Team

---

## Overview

This system automates the end-to-end workflow for research paper analysis using a modular, multi-agent architecture. It leverages fine-tuned language models, retrieval-augmented generation (RAG), and robust evaluation to deliver high-quality, context-aware summaries and bibliography management.

### Key Features
- **Multi-Agent System**: Specialized agents for extraction, planning, summarization, and bibliography.
- **Fine-Tuned LLMs**: Domain-adapted models for academic text.
- **RAG Integration**: Contextual retrieval from a local vector store.
- **Extensible & Modular**: Easily add new agents or processing steps.
- **CLI & Web UI**: Multiple interfaces for diverse user needs.

---

## Architecture

### Components
- **Orchestrator**: Coordinates agent workflow and error handling.
- **Extractor Agent**: Extracts and preprocesses text from PDFs.
- **Planner Agent**: Analyzes structure and proposes summarization strategy.
- **Summarizer Agent**: Generates structured or unstructured summaries using LLMs (supports RAG).
- **Bibliography Agent**: Extracts, formats, and enriches citations.
- **Vector Store**: Stores embeddings for RAG (ChromaDB, persistent, local).

### Data Flow
1. **PDF Upload** → Extractor Agent → Cleaned Text
2. **Text** → Planner Agent → Processing Strategy
3. **Text + Strategy (+ RAG Context)** → Summarizer Agent → Summary
4. **Text + Summary** → Bibliography Agent → Citations
5. **Output**: JSON summary, citations, and metadata

---

## Design Decisions

### Multi-Agent Pattern
- **Why**: Separation of concerns, easier maintenance, parallelism.
- **Alternatives**: Monolith (rejected: hard to maintain), microservices (overkill), plugin system (unnecessary).

### Fine-Tuning with LoRA
- **Why**: Efficient adaptation to academic domain, low resource usage.
- **Alternatives**: Full fine-tuning (too costly), prompt engineering only (limited), adapter layers (more complex).

### RAG with ChromaDB
- **Why**: Fast, local, persistent, easy integration.
- **Alternatives**: Pinecone (cloud, paid), FAISS (no persistence).

---

## Implementation Details

### Code Structure
- `main.py`: CLI entry point, argument parsing, command dispatch
- `src/agents/`: Agent implementations (extractor, planner, summarizer, bibliography)
- `src/models/`: Fine-tuning, evaluation, and utilities
- `data/`: Training and evaluation datasets

### Model Usage
- Default summarizer: `facebook/bart-large-cnn` (or path to fine-tuned model)
- Supports LoRA adapters for efficient domain adaptation
- RAG context is passed as supporting/background only

### CLI Examples
- **Single Paper:** `python main.py process --input paper.pdf --output summary.json --model ./finetuned_model`
- **Batch:** `python main.py batch papers/ --output results/ --model ./finetuned_model`
- **Fine-Tuning:** `python main.py fine-tune --train-data data.json --output ./finetuned_model --epochs 5`
- **Evaluation:** `python main.py evaluate --test-data eval.json --model ./finetuned_model`

---

## System Diagrams

### Agent Workflow
```
PDF → Extractor → Planner → Summarizer (+RAG) → Bibliography → Output
```

### Orchestrator Sequence
```
User → Web/CLI → Orchestrator → [Extractor → Planner → Summarizer → Bibliography] → Output
```

---

## Evaluation & Metrics
- **Summary Quality**: ROUGE-1, ROUGE-L, BLEU
- **Processing Speed**: < 3 minutes per paper
- **Memory Usage**: < 4GB RAM (with quantization)
- **Success Rate**: > 90% successful processing

---

## Extending the System
- Add new agents by subclassing the base agent interface
- Swap out LLMs by changing the model path in config/CLI
- Integrate new vector stores or citation formats as needed

---

## Action Items / TODOs
- [ ] Expand training dataset for better fine-tuning
- [ ] Add more robust error handling/logging
- [ ] Enhance RAG to support cross-paper citation linking
- [ ] Improve Web UI for batch uploads and visualization

---

For further details, see code comments and individual agent modules.