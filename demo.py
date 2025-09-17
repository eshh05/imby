"""
Demo script to showcase the Research Paper Summarization AI Agent capabilities.
"""

import asyncio
import json
import os
from pathlib import Path
from fpdf import FPDF
import tempfile
from src.agents.orchestrator import PaperSummarizationOrchestrator
from src.evaluation.metrics import SummarizationEvaluator
from src.models.fine_tuning import AcademicSummarizationFineTuner

def create_sample_paper():
    """Create a sample research paper for demonstration."""
    
    sample_paper = """
Abstract

This paper presents a novel approach to neural machine translation using transformer architectures with enhanced attention mechanisms. We propose a sparse attention pattern that reduces computational complexity while maintaining translation quality. Our method achieves state-of-the-art results on WMT datasets, improving BLEU scores by 2.3 points over previous best methods while reducing training time by 40%.

1. Introduction

Neural machine translation (NMT) has revolutionized the field of computational linguistics, largely replacing traditional statistical methods. The introduction of the transformer architecture by Vaswani et al. (2017) marked a significant milestone, enabling models to capture long-range dependencies more effectively than recurrent neural networks.

However, the quadratic complexity of self-attention mechanisms in transformers poses scalability challenges for longer sequences. Recent work has explored various approaches to address this limitation, including sparse attention patterns, local attention windows, and hierarchical attention structures.

2. Methodology

Our approach builds upon the standard transformer architecture but introduces a novel sparse attention pattern that we call "Structured Sparse Attention" (SSA). The key innovation lies in selectively attending to only the most relevant tokens while maintaining the model's ability to capture long-range dependencies.

2.1 Structured Sparse Attention

The SSA mechanism works by:
1. Computing attention scores for all token pairs
2. Applying a learned sparsity mask to retain only high-importance connections
3. Normalizing the sparse attention weights
4. Computing the weighted sum of values

This approach reduces the effective attention complexity from O(n²) to O(n log n) while preserving model performance.

2.2 Training Procedure

We train our models using the Adam optimizer with a learning rate schedule that includes warmup and cosine decay. The sparsity masks are learned jointly with the main translation objective using a multi-task learning framework.

3. Experiments

We evaluate our method on several WMT translation tasks, including English-German, English-French, and English-Chinese translation pairs. Our experimental setup follows standard practices in the field.

3.1 Datasets

- WMT'14 English-German: 4.5M sentence pairs
- WMT'14 English-French: 36M sentence pairs  
- WMT'17 English-Chinese: 20M sentence pairs

3.2 Baselines

We compare against several strong baselines:
- Transformer Base (Vaswani et al., 2017)
- Transformer Big (Vaswani et al., 2017)
- Linformer (Wang et al., 2020)
- Performer (Choromanski et al., 2021)

4. Results

Our Structured Sparse Attention method achieves significant improvements across all translation tasks:

English-German: 28.4 BLEU (+2.1 over Transformer Big)
English-French: 41.0 BLEU (+1.8 over Transformer Big)
English-Chinese: 24.2 BLEU (+2.7 over Transformer Big)

Additionally, training time is reduced by an average of 40% compared to standard transformers, making the approach practically attractive for large-scale deployment.

4.1 Ablation Studies

We conduct extensive ablation studies to understand the contribution of different components:
- Sparsity pattern design: +1.2 BLEU
- Multi-task learning objective: +0.8 BLEU
- Attention head specialization: +0.3 BLEU

5. Analysis

Our analysis reveals that the learned sparsity patterns exhibit interesting linguistic properties. The model tends to focus on syntactically related tokens and maintains strong connections between content words while pruning connections to function words.

Visualization of attention patterns shows that our method preserves the ability to model long-range dependencies while significantly reducing the number of attention computations.

6. Conclusion

We have presented a novel sparse attention mechanism for neural machine translation that achieves superior performance while reducing computational requirements. The Structured Sparse Attention approach offers a promising direction for scaling transformer models to longer sequences and larger datasets.

Future work will explore applications to other sequence-to-sequence tasks and investigate adaptive sparsity patterns that can adjust based on input complexity.

References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020). Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768.

Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., ... & Weller, A. (2021). Rethinking attention with performers. In International Conference on Learning Representations.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. In Advances in neural information processing systems (pp. 1877-1901).

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
"""
    
    return sample_paper

def create_pdf_from_text(text, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(0, 10, line, ln=True)
    pdf.output(output_path)

async def run_demo():
    """Run a complete demonstration of the system."""
    
    print("🤖 Research Paper Summarization AI Agent - Demo")
    print("=" * 60)
    
    # Step 1: Create sample paper
    print("\n📄 Step 1: Creating sample research paper...")
    sample_paper = create_sample_paper()
    
    # Save to temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        temp_file_path = f.name
    create_pdf_from_text(sample_paper, temp_file_path)
    print(f"✅ Sample paper created as PDF ({len(sample_paper)} characters)")
    
    try:
        # Step 2: Initialize the orchestrator
        print("\n🔧 Step 2: Initializing AI agents...")
        orchestrator = PaperSummarizationOrchestrator()
        print("✅ Multi-agent system initialized")
        
        # Step 3: Process the paper
        print("\n⚡ Step 3: Processing paper through AI pipeline...")
        print("   - Extracting content...")
        print("   - Planning processing strategy...")
        print("   - Generating summary with fine-tuned model...")
        print("   - Extracting citations and bibliography...")
        
        result = await orchestrator.process_paper({
            'file_path': temp_file_path
        })
        
        if result['success']:
            print("✅ Processing completed successfully!")
            
            # Step 4: Display results
            print("\n📋 Step 4: Generated Summary")
            print("-" * 40)
            
            summary = result['summary']['summary']
            print(f"Summary Type: {summary['type']}")
            print(f"Summary Length: {len(summary['full_text'])} characters")
            print("\nGenerated Summary:")
            print(summary['full_text'])
            
            # Display key points
            if 'key_points' in summary:
                print("\n🔑 Key Points:")
                for i, point in enumerate(summary['key_points'], 1):
                    print(f"{i}. {point}")
            
            # Step 5: Display bibliography
            print("\n📚 Step 5: Bibliography Analysis")
            print("-" * 40)
            
            bibliography = result['bibliography']
            print(f"Citations Found: {bibliography['citation_count']}")
            
            if bibliography['citations']:
                print("\nExtracted Citations:")
                for citation in bibliography['citations'][:5]:  # Show first 5
                    print(f"• {citation['text']} (Type: {citation['type']})")
                
                # Show formatted bibliography
                if 'apa' in bibliography['bibliographies']:
                    print("\nAPA Format Bibliography:")
                    for citation in bibliography['bibliographies']['apa'][:3]:
                        print(f"• {citation}")
            
            # Step 6: Processing metadata
            print("\n📊 Step 6: Processing Metadata")
            print("-" * 40)
            
            metadata = result['processing_metadata']
            print(f"Original Length: {metadata.get('text_length', 0):,} characters")
            print(f"Summary Length: {metadata.get('summary_length', 0):,} characters")
            print(f"Compression Ratio: {metadata.get('compression_ratio', 0):.1%}")
            print(f"Citations Found: {metadata.get('citation_count', 0)}")
            
            # Step 7: Evaluation
            print("\n🎯 Step 7: Quality Evaluation")
            print("-" * 40)
            
            evaluator = SummarizationEvaluator()
            
            # Create a reference summary for evaluation
            reference_summary = """
            This paper introduces Structured Sparse Attention (SSA) for neural machine translation, 
            which reduces attention complexity from O(n²) to O(n log n) while maintaining performance. 
            The method achieves state-of-the-art BLEU scores on WMT datasets with 2.3 point improvements 
            and 40% faster training. The approach uses learned sparsity masks to selectively attend to 
            relevant tokens while preserving long-range dependencies.
            """
            
            evaluation = evaluator.evaluate_summary(
                generated_summary=summary['full_text'],
                reference_summary=reference_summary,
                source_text=sample_paper
            )
            
            print(f"ROUGE-1 F1: {evaluation['rouge']['rouge1_f']:.3f}")
            print(f"ROUGE-2 F1: {evaluation['rouge']['rouge2_f']:.3f}")
            print(f"ROUGE-L F1: {evaluation['rouge']['rougeL_f']:.3f}")
            print(f"Semantic Similarity: {evaluation['semantic_similarity']['overall_similarity']:.3f}")
            print(f"Overall Quality Score: {evaluation['overall_score']:.3f}")
            
            # Step 8: Save results
            print("\n💾 Step 8: Saving Results")
            print("-" * 40)
            
            output_file = "demo_results.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'processing_result': result,
                    'evaluation_metrics': evaluation
                }, f, indent=2, default=str)
            
            print(f"✅ Results saved to: {output_file}")
            
            # Step 9: Demo fine-tuning (optional)
            print("\n🔧 Step 9: Fine-tuning Demo (Creating Sample Dataset)")
            print("-" * 40)
            
            fine_tuner = AcademicSummarizationFineTuner({
                'base_model': 'microsoft/DialoGPT-medium',
                'output_dir': 'models/demo_lora'
            })
            
            dataset_path = 'data/training/demo_dataset.json'
            fine_tuner.create_sample_dataset(dataset_path)
            print(f"✅ Sample training dataset created: {dataset_path}")
            print("   (Run 'python main.py fine-tune' to start actual fine-tuning)")
            
        else:
            print(f"❌ Processing failed: {result.get('error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        return 1
        
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext Steps:")
    print("1. Run 'streamlit run app.py' for the web interface")
    print("2. Try 'python main.py process your_paper.pdf' with your own papers")
    print("3. Run 'python main.py fine-tune' to fine-tune on your data")
    print("4. Explore the evaluation metrics with 'python main.py evaluate'")
    
    return 0

def main():
    """Main demo entry point."""
    return asyncio.run(run_demo())

if __name__ == "__main__":
    exit(main())
