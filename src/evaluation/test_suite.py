"""
Test suite for evaluating the research paper summarization system.
"""

import asyncio
import json
import os
from typing import Dict, Any, List
import logging
from ..agents.orchestrator import PaperSummarizationOrchestrator
from .metrics import SummarizationEvaluator, BatchEvaluator

class SummarizationTestSuite:
    """Test suite for comprehensive evaluation of the summarization system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("test_suite")
        
        # Initialize components
        self.orchestrator = PaperSummarizationOrchestrator(config.get('orchestrator', {}))
        self.evaluator = SummarizationEvaluator(config.get('evaluator', {}))
        self.batch_evaluator = BatchEvaluator(self.evaluator)
        
        # Test data paths
        self.test_data_dir = config.get('test_data_dir', 'data/test')
        self.results_dir = config.get('results_dir', 'results/evaluation')
        
        # Ensure directories exist
        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        
        self.logger.info("Starting full evaluation suite")
        
        results = {
            'test_summary': {},
            'individual_tests': {},
            'performance_metrics': {},
            'error_analysis': {}
        }
        
        # Test 1: Single paper processing
        self.logger.info("Running single paper test")
        single_paper_results = await self._test_single_paper_processing()
        results['individual_tests']['single_paper'] = single_paper_results
        
        # Test 2: Batch processing
        self.logger.info("Running batch processing test")
        batch_results = await self._test_batch_processing()
        results['individual_tests']['batch_processing'] = batch_results
        
        # Test 3: Different paper types
        self.logger.info("Running paper type variation test")
        paper_type_results = await self._test_different_paper_types()
        results['individual_tests']['paper_types'] = paper_type_results
        
        # Test 4: Performance benchmarks
        self.logger.info("Running performance benchmarks")
        performance_results = await self._test_performance_benchmarks()
        results['performance_metrics'] = performance_results
        
        # Test 5: Error handling
        self.logger.info("Running error handling test")
        error_results = await self._test_error_handling()
        results['error_analysis'] = error_results
        
        # Generate summary
        results['test_summary'] = self._generate_test_summary(results)
        
        # Save results
        self._save_results(results)
        
        self.logger.info("Full evaluation suite completed")
        return results
    
    async def _test_single_paper_processing(self) -> Dict[str, Any]:
        """Test processing of a single paper."""
        
        # Create test paper
        test_paper = self._create_test_paper()
        
        try:
            # Process paper
            result = await self.orchestrator.process_paper({
                'file_content': test_paper['content'].encode(),
                'filename': 'test_paper.txt'
            })
            
            if result['success']:
                # Evaluate against reference
                evaluation = self.evaluator.evaluate_summary(
                    generated_summary=result['summary']['full_text'],
                    reference_summary=test_paper['reference_summary'],
                    source_text=test_paper['content']
                )
                
                return {
                    'success': True,
                    'processing_result': result,
                    'evaluation_metrics': evaluation,
                    'processing_time': result.get('processing_metadata', {}).get('processing_time', 0)
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'stage': result.get('stage', 'unknown')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stage': 'exception'
            }
    
    async def _test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing of multiple papers."""
        
        # Create multiple test papers
        test_papers = [self._create_test_paper(i) for i in range(3)]
        
        try:
            # Process batch
            batch_input = [
                {
                    'file_content': paper['content'].encode(),
                    'filename': f'test_paper_{i}.txt'
                }
                for i, paper in enumerate(test_papers)
            ]
            
            results = await self.orchestrator.process_batch(batch_input)
            
            # Add reference summaries for evaluation
            for i, result in enumerate(results):
                if result['success']:
                    result['reference_summary'] = test_papers[i]['reference_summary']
                    result['source_text'] = test_papers[i]['content']
            
            # Evaluate batch
            batch_evaluation = self.batch_evaluator.evaluate_batch(results)
            
            return {
                'success': True,
                'batch_results': results,
                'batch_evaluation': batch_evaluation,
                'papers_processed': len(results),
                'success_rate': sum(1 for r in results if r['success']) / len(results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_different_paper_types(self) -> Dict[str, Any]:
        """Test processing different types of academic papers."""
        
        paper_types = [
            'computer_science',
            'biology',
            'physics',
            'economics'
        ]
        
        results = {}
        
        for paper_type in paper_types:
            test_paper = self._create_test_paper_by_type(paper_type)
            
            try:
                result = await self.orchestrator.process_paper({
                    'file_content': test_paper['content'].encode(),
                    'filename': f'{paper_type}_paper.txt'
                })
                
                if result['success']:
                    evaluation = self.evaluator.evaluate_summary(
                        generated_summary=result['summary']['full_text'],
                        reference_summary=test_paper['reference_summary'],
                        source_text=test_paper['content']
                    )
                    
                    results[paper_type] = {
                        'success': True,
                        'evaluation': evaluation,
                        'summary_length': len(result['summary']['full_text']),
                        'sections_identified': len(result['summary']['sections'])
                    }
                else:
                    results[paper_type] = {
                        'success': False,
                        'error': result.get('error')
                    }
                    
            except Exception as e:
                results[paper_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    async def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks."""
        
        import time
        
        # Test processing speed
        test_paper = self._create_test_paper()
        
        start_time = time.time()
        result = await self.orchestrator.process_paper({
            'file_content': test_paper['content'].encode(),
            'filename': 'benchmark_paper.txt'
        })
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Memory usage (simplified)
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'processing_time_seconds': processing_time,
            'memory_usage_mb': memory_usage,
            'words_per_second': len(test_paper['content'].split()) / processing_time if processing_time > 0 else 0,
            'success': result['success'] if result else False
        }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling capabilities."""
        
        error_tests = {
            'empty_file': {'file_content': b'', 'filename': 'empty.txt'},
            'invalid_pdf': {'file_content': b'invalid pdf content', 'filename': 'invalid.pdf'},
            'very_long_text': {
                'file_content': ('A' * 100000).encode(), 
                'filename': 'long.txt'
            }
        }
        
        results = {}
        
        for test_name, test_input in error_tests.items():
            try:
                result = await self.orchestrator.process_paper(test_input)
                
                results[test_name] = {
                    'handled_gracefully': not result['success'],
                    'error_message': result.get('error', 'No error message'),
                    'stage': result.get('stage', 'unknown')
                }
                
            except Exception as e:
                results[test_name] = {
                    'handled_gracefully': False,
                    'exception': str(e)
                }
        
        return results
    
    def _create_test_paper(self, index: int = 0) -> Dict[str, Any]:
        """Create a test paper with known content and reference summary."""
        
        content = f"""
        Abstract: This paper presents a novel approach to machine learning optimization using gradient descent variants. 
        We propose several improvements that result in faster convergence and better generalization.
        
        Introduction: Machine learning optimization has been a central challenge in the field. Traditional gradient 
        descent methods often suffer from slow convergence and local minima problems. Recent advances in adaptive 
        learning rates have shown promise in addressing these issues.
        
        Methodology: We implement a modified Adam optimizer with momentum scheduling and learning rate decay. 
        Our approach combines the benefits of adaptive learning rates with momentum-based optimization to achieve 
        better performance on various machine learning tasks.
        
        Results: Experimental results on benchmark datasets show that our method achieves 15% faster convergence 
        compared to standard optimizers while maintaining comparable or better final accuracy. The method shows 
        particular improvements on deep neural networks and convolutional architectures.
        
        Conclusion: The proposed optimization method offers a practical solution for faster training of machine 
        learning models. Future work will explore applications to transformer architectures and reinforcement learning.
        
        References:
        [1] Smith, J. (2020). Optimization in Deep Learning. Journal of ML Research.
        [2] Johnson, A. et al. (2021). Adaptive Learning Rates. Conference on Neural Networks.
        """
        
        reference_summary = """
        This paper introduces an improved machine learning optimizer that combines Adam optimization with momentum 
        scheduling and learning rate decay. The method achieves 15% faster convergence than standard optimizers 
        while maintaining accuracy, with particular benefits for deep neural networks and CNNs. The approach 
        addresses traditional gradient descent limitations of slow convergence and local minima through adaptive 
        learning rates and momentum-based optimization.
        """
        
        return {
            'content': content,
            'reference_summary': reference_summary,
            'metadata': {
                'title': f'Test Paper {index}',
                'authors': ['Test Author'],
                'year': 2024
            }
        }
    
    def _create_test_paper_by_type(self, paper_type: str) -> Dict[str, Any]:
        """Create test papers for different academic domains."""
        
        papers = {
            'computer_science': {
                'content': """
                Abstract: We present a new algorithm for distributed computing that improves load balancing 
                across multiple nodes. Our approach reduces communication overhead by 30%.
                
                Introduction: Distributed systems face challenges in load balancing and fault tolerance.
                
                Methodology: We implement a consensus-based load balancing protocol using Raft consensus.
                
                Results: Performance tests show significant improvements in throughput and latency.
                
                Conclusion: The proposed method offers better scalability for distributed applications.
                """,
                'reference_summary': 'A new distributed computing algorithm improves load balancing and reduces communication overhead by 30% using Raft consensus protocol.'
            },
            'biology': {
                'content': """
                Abstract: This study investigates protein folding mechanisms in Alzheimer's disease using 
                molecular dynamics simulations. We identify key structural changes in amyloid beta proteins.
                
                Introduction: Protein misfolding is implicated in neurodegenerative diseases.
                
                Methodology: We used molecular dynamics simulations with CHARMM force field.
                
                Results: Simulations reveal critical folding intermediates and aggregation pathways.
                
                Conclusion: Understanding these mechanisms may lead to therapeutic interventions.
                """,
                'reference_summary': 'Molecular dynamics simulations reveal protein folding mechanisms in Alzheimer\'s disease, identifying key structural changes in amyloid beta proteins that could inform therapeutic approaches.'
            },
            'physics': {
                'content': """
                Abstract: We report measurements of quantum entanglement in a superconducting qubit system. 
                Our results demonstrate high-fidelity two-qubit gates with 99.5% accuracy.
                
                Introduction: Quantum computing requires high-fidelity quantum gates for practical applications.
                
                Methodology: We fabricated superconducting qubits using Josephson junctions.
                
                Results: Gate fidelities exceed 99% with coherence times of 100 microseconds.
                
                Conclusion: These results represent progress toward fault-tolerant quantum computing.
                """,
                'reference_summary': 'High-fidelity quantum gates in superconducting qubits achieve 99.5% accuracy with 100 microsecond coherence times, advancing fault-tolerant quantum computing.'
            },
            'economics': {
                'content': """
                Abstract: This paper analyzes the impact of monetary policy on inflation expectations using 
                a dynamic stochastic general equilibrium model. We find significant effects on long-term expectations.
                
                Introduction: Central bank policy affects inflation through various transmission mechanisms.
                
                Methodology: We employ a DSGE model with rational expectations and sticky prices.
                
                Results: Policy announcements have persistent effects on inflation expectations.
                
                Conclusion: Central banks should consider expectation management in policy design.
                """,
                'reference_summary': 'DSGE model analysis shows monetary policy significantly affects long-term inflation expectations, suggesting central banks should incorporate expectation management in policy design.'
            }
        }
        
        return papers.get(paper_type, self._create_test_paper())
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results."""
        
        summary = {
            'overall_success': True,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'key_metrics': {}
        }
        
        # Count tests
        for test_category, test_results in results['individual_tests'].items():
            if isinstance(test_results, dict):
                if test_results.get('success', False):
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                    summary['overall_success'] = False
                summary['total_tests'] += 1
        
        # Extract key metrics
        if 'single_paper' in results['individual_tests']:
            single_result = results['individual_tests']['single_paper']
            if single_result.get('success') and 'evaluation_metrics' in single_result:
                eval_metrics = single_result['evaluation_metrics']
                summary['key_metrics']['rouge_score'] = eval_metrics.get('rouge', {}).get('rouge1_f', 0)
                summary['key_metrics']['semantic_similarity'] = eval_metrics.get('semantic_similarity', {}).get('overall_similarity', 0)
                summary['key_metrics']['overall_score'] = eval_metrics.get('overall_score', 0)
        
        if 'batch_processing' in results['individual_tests']:
            batch_result = results['individual_tests']['batch_processing']
            if batch_result.get('success'):
                summary['key_metrics']['batch_success_rate'] = batch_result.get('success_rate', 0)
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Test results saved to {filepath}")

async def main():
    """Run the test suite."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize test suite
    test_suite = SummarizationTestSuite()
    
    # Run evaluation
    results = await test_suite.run_full_evaluation()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    summary = results['test_summary']
    print(f"Overall Success: {summary['overall_success']}")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
    
    if 'key_metrics' in summary:
        print("\nKey Metrics:")
        for metric, value in summary['key_metrics'].items():
            print(f"  {metric}: {value:.3f}")
    
    print("\nDetailed results saved to results/evaluation/")

if __name__ == "__main__":
    asyncio.run(main())
