"""
Orchestrator - Coordinates multi-agent workflow for paper summarization.
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from .base_agent import BaseAgent, AgentResult
from .planner_agent import PlannerAgent
from .extractor_agent import ExtractorAgent
from .summarizer_agent import SummarizerAgent
from .bibliography_agent import BibliographyAgent

class PaperSummarizationOrchestrator:
    """Orchestrates the multi-agent  workflow for research paper summarization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("orchestrator")
        
        # Initialize agents
        self.planner = PlannerAgent(self.config.get('planner', {}))
        self.logger.info("Planner Agent Initalized")
        self.extractor = ExtractorAgent(self.config.get('extractor', {}))
        self.summarizer = SummarizerAgent(self.config.get('summarizer', {}))
        self.bibliography = BibliographyAgent(self.config.get('bibliography', {}))
        
    async def process_paper(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research paper through the complete multi-agent pipeline.
        Args:
            input_data: Dict containing either 'file_path' or 'file_content' and optional 'rag_context'
        Returns:
            Complete processing results including summary and bibliography
        """
        try:
            self.logger.info("Starting paper processing pipeline")
            rag_context = input_data.get('rag_context')
            if rag_context:
                self.logger.info(f"RAG context received with {len(rag_context)} items: {[c['id'] for c in rag_context]}")
            # Step 1: Extract content from PDF
            self.logger.info("Step 1: Extracting content from PDF")
            extraction_result = await self.extractor.execute(input_data)
            if not extraction_result.success:
                return {
                    'success': False,
                    'error': f"Content extraction failed: {extraction_result.error}",
                    'stage': 'extraction'
                }
            extracted_text = extraction_result.data['text']
            metadata = extraction_result.data['metadata']
            # Step 2: Create processing plan
            self.logger.info("Step 2: Creating processing plan")
            plan_result = await self.planner.execute({'text': extracted_text})
            if not plan_result.success:
                return {
                    'success': False,
                    'error': f"Planning failed: {plan_result.error}",
                    'stage': 'planning'
                }
            processing_plan = plan_result.data['plan']
            # Step 3: Generate summary
            self.logger.info("Step 3: Generating summary")
            summary_input = {
                'text': extracted_text,
                'plan': processing_plan
            }
            if rag_context:
                summary_input['rag_context'] = rag_context
            summary_result = await self.summarizer.execute(summary_input)
            if not summary_result.success:
                return {
                    'success': False,
                    'error': f"Summarization failed: {summary_result.error}",
                    'stage': 'summarization'
                }
            # Step 4: Process bibliography
            self.logger.info("Step 4: Processing bibliography")
            citation_plan = processing_plan['strategy']['citation_extraction']
            bibliography_result = await self.bibliography.execute({
                'text': extracted_text,
                'citation_plan': citation_plan
            })
            if not bibliography_result.success:
                self.logger.warning(f"Bibliography processing failed: {bibliography_result.error}")
                # Continue without bibliography rather than failing completely
                bibliography_data = {
                    'citations': [],
                    'bibliographies': {},
                    'citation_count': 0
                }
            else:
                bibliography_data = bibliography_result.data
            # Combine results
            final_result = {
                'success': True,
                'paper_metadata': metadata,
                'processing_plan': processing_plan,
                'summary': summary_result.data,
                'bibliography': bibliography_data,
                'processing_metadata': {
                    'extraction_method': extraction_result.metadata.get('extraction_method'),
                    'text_length': extraction_result.metadata.get('text_length'),
                    'summary_length': summary_result.metadata.get('summary_length'),
                    'compression_ratio': summary_result.metadata.get('compression_ratio'),
                    'citation_count': bibliography_data['citation_count']
                }
            }
            self.logger.info("Paper processing completed successfully")
            return final_result
        except Exception as e:
            error_msg = f"Orchestrator error: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'stage': 'orchestration'
            }
    
    async def process_batch(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple papers in parallel."""
        self.logger.info(f"Starting batch processing of {len(papers)} papers")
        
        # Process papers concurrently
        tasks = [self.process_paper(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': f"Exception in paper {i}: {str(result)}",
                    'paper_index': i
                })
            else:
                result['paper_index'] = i
                processed_results.append(result)
        
        self.logger.info(f"Batch processing completed: {sum(1 for r in processed_results if r['success'])} successful")
        return processed_results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status information about all agents."""
        return {
            'planner': {
                'name': self.planner.name,
                'config': self.planner.config
            },
            'extractor': {
                'name': self.extractor.name,
                'extraction_method': self.extractor.extraction_method
            },
            'summarizer': {
                'name': self.summarizer.name,
                'model_name': self.summarizer.model_name,
                'model_loaded': self.summarizer.model is not None
            },
            'bibliography': {
                'name': self.bibliography.name,
                'supported_styles': self.bibliography.citation_styles
            }
        }
