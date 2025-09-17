"""
Command-line interface for the Research Paper Summarization AI Agent.
"""

import asyncio
import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.agents.orchestrator import PaperSummarizationOrchestrator
from src.evaluation.test_suite import SummarizationTestSuite
from src.models.fine_tuning import AcademicSummarizationFineTuner
from src.rag.vector_store import PaperVectorStore

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def process_single_paper(args):
    """Process a single research paper."""
    
    print(f"Processing paper: {args.input}")
    
    config = {
        'extractor': {'method': args.extraction_method},
        'summarizer': {'model_name': args.model}
    }
    
    orchestrator = PaperSummarizationOrchestrator(config)
    vector_store = PaperVectorStore()
    
    # Extract text for RAG retrieval
    extraction_result = await orchestrator.extractor.execute({'file_path': args.input})
    if not extraction_result.success:
        print(f"❌ Extraction failed: {extraction_result.error}")
        return 1
    extracted_text = extraction_result.data['text']
    rag_context = vector_store.search_similar_papers(extracted_text, n_results=3)
    print(f"RAG retrieved {len(rag_context)} contexts: {[c['id'] for c in rag_context]}")
    
    # Process paper (always use RAG)
    result = await orchestrator.process_paper({'file_path': args.input, 'rag_context': rag_context})
    
    if result['success']:
        print("✅ Processing completed successfully!")
        
        # Save results
        output_path = args.output or f"{Path(args.input).stem}_summary.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {output_path}")
        
        # Display summary
        if args.verbose:
            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(result['summary']['full_text'])
            
            print("\n" + "="*50)
            print("CITATIONS")
            print("="*50)
            for citation in result['bibliography']['citations']:
                print(f"• {citation['text']}")
        
    else:
        print(f"❌ Processing failed: {result.get('error')}")
        return 1
    
    return 0

async def process_batch(args):
    """Process multiple papers in batch."""
    
    input_dir = Path(args.batch)
    
    if not input_dir.is_dir():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
    
    # Find all PDF and TXT files
    paper_files = []
    for ext in ['*.pdf', '*.txt']:
        paper_files.extend(input_dir.glob(ext))
    
    if not paper_files:
        print(f"❌ No papers found in: {input_dir}")
        return 1
    
    print(f"📚 Found {len(paper_files)} papers to process")
    
    config = {
        'extractor': {'method': args.extraction_method},
        'summarizer': {'model_name': args.model}
    }
    
    orchestrator = PaperSummarizationOrchestrator(config)
    vector_store = PaperVectorStore()
    
    batch_input = []
    for file_path in paper_files:
        extraction_result = await orchestrator.extractor.execute({'file_path': str(file_path)})
        if extraction_result.success:
            extracted_text = extraction_result.data['text']
            rag_context = vector_store.search_similar_papers(extracted_text, n_results=3)
            print(f"RAG for {file_path.name}: {len(rag_context)} contexts: {[c['id'] for c in rag_context]}")
            batch_input.append({'file_path': str(file_path), 'rag_context': rag_context})
        else:
            print(f"❌ Extraction failed for {file_path.name}: {extraction_result.error}")
            batch_input.append({'file_path': str(file_path), 'rag_context': []})
    
    # Process batch
    results = await orchestrator.process_batch(batch_input)
    
    # Save results
    output_dir = Path(args.output) if args.output else input_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, (result, file_path) in enumerate(zip(results, paper_files)):
        if result['success']:
            successful += 1
            output_file = output_dir / f"{file_path.stem}_summary.json"
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"✅ {file_path.name} -> {output_file}")
        else:
            failed += 1
            print(f"❌ {file_path.name}: {result.get('error')}")
    
    print(f"\n📊 Batch processing completed: {successful} successful, {failed} failed")
    return 0

async def run_evaluation(args):
    """Run the evaluation test suite."""
    
    print("🧪 Running evaluation test suite...")
    
    config = {
        'test_data_dir': args.test_data or 'data/test',
        'results_dir': args.output or 'results/evaluation'
    }
    
    test_suite = SummarizationTestSuite(config)
    results = await test_suite.run_full_evaluation()
    
    # Display summary
    summary = results['test_summary']
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Success: {summary['overall_success']}")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
    
    if 'key_metrics' in summary:
        print("\nKey Metrics:")
        for metric, value in summary['key_metrics'].items():
            print(f"  {metric}: {value:.3f}")
    
    return 0 if summary['overall_success'] else 1

def run_fine_tuning(args):
    """Run the fine-tuning pipeline."""
    
    print("🔧 Starting fine-tuning pipeline...")
    
    config = {
        'base_model': args.model,
        'output_dir': args.output or 'models/summarizer_lora',
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    fine_tuner = AcademicSummarizationFineTuner(config)
    
    # Create sample dataset if no training data provided
    # if not args.train_data:
    #     print("📝 No training data provided, creating sample dataset...")
    #     dataset_path = 'data/training/sample_academic_summaries.json'
    #     fine_tuner.create_sample_dataset(dataset_path)
    #     args.train_data = dataset_path
    
    # Prepare model
    fine_tuner.prepare_model()
    
    # Prepare dataset
    train_dataset = fine_tuner.prepare_dataset(args.train_data)
    
    # Split for evaluation if requested
    eval_dataset = None
    if args.eval_split > 0:
        import torch.utils.data
        train_size = int((1 - args.eval_split) * len(train_dataset))
        eval_size = len(train_dataset) - train_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size]
        )
    
    # Train
    fine_tuner.train(train_dataset, eval_dataset)
    
    print(f"✅ Fine-tuning completed! Model saved to: {config['output_dir']}")
    return 0

def setup_vector_store(args):
    """Setup and populate vector store."""
    
    print("🗄️ Setting up vector store...")
    
    vector_store = PaperVectorStore({
        'persist_directory': args.output or 'data/vector_store'
    })
    
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_file() and input_path.suffix == '.json':
            # Load papers from JSON file
            with open(input_path) as f:
                papers_data = json.load(f)
            
            added_count = vector_store.add_papers_batch(papers_data)
            print(f"✅ Added {added_count} papers to vector store")
            
        elif input_path.is_dir():
            # Process all JSON files in directory
            json_files = list(input_path.glob('*.json'))
            
            total_added = 0
            for json_file in json_files:
                with open(json_file) as f:
                    papers_data = json.load(f)
                
                if isinstance(papers_data, list):
                    added_count = vector_store.add_papers_batch(papers_data)
                else:
                    # Single paper
                    success = vector_store.add_paper(
                        paper_id=json_file.stem,
                        content=papers_data.get('content', ''),
                        metadata=papers_data.get('metadata', {})
                    )
                    added_count = 1 if success else 0
                
                total_added += added_count
                print(f"📄 Processed {json_file.name}: {added_count} papers added")
            
            print(f"✅ Total papers added to vector store: {total_added}")
    
    # Display statistics
    stats = vector_store.get_collection_stats()
    print(f"📊 Vector store statistics: {stats}")
    
    return 0

def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Research Paper Summarization AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single paper
  python main.py process paper.pdf --output summary.json
  
  # Process multiple papers
  python main.py batch papers/ --output results/
  
  # Run evaluation
  python main.py evaluate --test-data data/test
  
  # Fine-tune model
  python main.py fine-tune --train-data data/training.json --epochs 3
  
  # Setup vector store
  python main.py vector-store --input papers.json --output data/vector_store
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--model', default='facebook/bart-large-cnn', help='Base model name')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process single paper
    process_parser = subparsers.add_parser('process', help='Process a single paper')
    process_parser.add_argument('input', help='Input paper file (PDF or TXT)')
    process_parser.add_argument('--output', '-o', help='Output file path')
    process_parser.add_argument('--extraction-method', choices=['pdfplumber', 'pymupdf', 'pypdf2'], 
                               default='pdfplumber', help='PDF extraction method')
    process_parser.add_argument('--model', default='facebook/bart-large-cnn', help='Base model name or path to fine-tuned model directory')

    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple papers')
    batch_parser.add_argument('batch', help='Input directory containing papers')
    batch_parser.add_argument('--output', '-o', help='Output directory')
    batch_parser.add_argument('--extraction-method', choices=['pdfplumber', 'pymupdf', 'pypdf2'], 
                             default='pdfplumber', help='PDF extraction method')
    
    # Evaluation
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation test suite')
    eval_parser.add_argument('--test-data', help='Test data directory')
    eval_parser.add_argument('--output', '-o', help='Results output directory')
    
    # Fine-tuning
    finetune_parser = subparsers.add_parser('fine-tune', help='Fine-tune the model')
    finetune_parser.add_argument('--train-data', help='Training data JSON file')
    finetune_parser.add_argument('--output', '-o', help='Model output directory')
    finetune_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    finetune_parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    finetune_parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    finetune_parser.add_argument('--eval-split', type=float, default=0.2, help='Evaluation split ratio')
    
    # Vector store setup
    vector_parser = subparsers.add_parser('vector-store', help='Setup vector store')
    vector_parser.add_argument('--input', '-i', help='Input papers (JSON file or directory)')
    vector_parser.add_argument('--output', '-o', help='Vector store directory')
    
    # Web interface
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=8501, help='Port number')
    web_parser.add_argument('--host', default='localhost', help='Host address')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'process':
            return asyncio.run(process_single_paper(args))
        elif args.command == 'batch':
            return asyncio.run(process_batch(args))
        elif args.command == 'evaluate':
            return asyncio.run(run_evaluation(args))
        elif args.command == 'fine-tune':
            return run_fine_tuning(args)
        elif args.command == 'vector-store':
            return setup_vector_store(args)
        elif args.command == 'web':
            import subprocess
            cmd = ['streamlit', 'run', 'app.py', '--server.port', str(args.port), '--server.address', args.host]
            subprocess.run(cmd)
            return 0
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
