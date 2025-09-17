"""
Summarizer Agent - Fine-tuned model for academic summarization.
"""

import torch
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import logging

class SummarizerAgent(BaseAgent):
    """Agent responsible for generating academic summaries using fine-tuned models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("summarizer", config)
        self.model_name = config.get('model_name', 'facebook/bart-large-cnn') if config else 'facebook/bart-large-cnn'
        self.max_length = config.get('max_length', 1024) if config else 1024
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Generate academic summary from paper content.
        
        Args:
            input_data: Dict containing 'text', optional 'plan', and optional 'rag_context'
            
        Returns:
            AgentResult with generated summary
        """
        try:
            if not self.validate_input(input_data, ['text']):
                return AgentResult(
                    success=False,
                    data={},
                    error="Missing required 'text' field"
                )
            
            # Load model if not already loaded
            if self.model is None:
                await self._load_model()
            
            text = input_data['text']
            plan = input_data.get('plan', {})
            rag_context = input_data.get('rag_context')
            context_texts = [c['content'] for c in rag_context if 'content' in c and c['content']] if rag_context else None
            if context_texts:
                self.logger.info(f"Summarizer received RAG context with {len(context_texts)} items.")
                self.logger.info(f"RAG context content samples: {[t[:1000] for t in context_texts]}")
            
            # Generate summary based on plan, pass RAG context
            summary = await self._generate_summary(text, plan, context_texts)
            
            result = AgentResult(
                success=True,
                data={
                    'summary': summary,
                    'summary_type': summary['type'],
                    'sections': summary['sections']
                },
                metadata={
                    'input_length': len(text),
                    'summary_length': len(summary['full_text']),
                    'compression_ratio': len(summary['full_text']) / len(text),
                    'model_used': self.model_name
                }
            )
            
            self.log_execution(input_data, result)
            return result
            
        except Exception as e:
            error_msg = f"Error in summarization: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data={},
                error=error_msg
            )
    
    async def _load_model(self):
        """Load the summarization model and tokenizer."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.model = "fallback"
            self.tokenizer = "fallback"
    
    async def _generate_summary(self, text: str, plan: Dict[str, Any], rag_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate summary based on text, processing plan, and optional RAG context."""
        
        if self.model == "fallback":
            return self._generate_extractive_summary(text, plan)
        
        # Determine summary approach based on plan
        approach = plan.get('strategy', {}).get('summarization_approach', 'unstructured_full_text')
        
        if approach == 'structured_with_abstract':
            return await self._generate_structured_summary(text, plan, rag_contexts)
        elif approach == 'structured_multi_section':
            return await self._generate_multi_section_summary(text, plan, rag_contexts)
        else:
            return await self._generate_full_text_summary(text, rag_contexts)
    
    async def _generate_structured_summary(self, text: str, plan: Dict[str, Any], rag_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate structured summary with identified sections and optional RAG context."""
        sections = {}
        
        # Extract abstract if available
        abstract = self._extract_section(text, 'abstract')
        if abstract:
            sections['abstract'] = await self._summarize_chunk(abstract, rag_contexts)
        
        # Generate summaries for key sections
        priority_sections = plan.get('strategy', {}).get('priority_sections', [])
        
        for section_type in priority_sections:
            if section_type != 'abstract':
                section_text = self._extract_section(text, section_type)
                if section_text:
                    sections[section_type] = await self._summarize_chunk(section_text, rag_contexts)
        
        # Combine into full summary
        full_summary = self._combine_section_summaries(sections)
        
        return {
            'type': 'structured',
            'sections': sections,
            'full_text': full_summary,
            'key_points': self._extract_key_points(sections)
        }
    
    async def _generate_multi_section_summary(self, text: str, plan: Dict[str, Any], rag_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate summary for papers with multiple clear sections and optional RAG context."""
        sections = {}
        identified_sections = plan.get('sections', [])
        
        for section_info in identified_sections:
            section_type = section_info['type']
            section_text = self._extract_section_by_position(text, section_info)
            
            if section_text and len(section_text.strip()) > 50:
                sections[section_type] = await self._summarize_chunk(section_text, rag_contexts)
        
        full_summary = self._combine_section_summaries(sections)
        
        return {
            'type': 'multi_section',
            'sections': sections,
            'full_text': full_summary,
            'key_points': self._extract_key_points(sections)
        }
    
    async def _generate_full_text_summary(self, text: str, rag_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate summary from full text with optional RAG context."""
        
        # Split text into chunks for processing
        chunks = self._split_text_into_chunks(text, max_chunk_size=2000)
        chunk_summaries = []
        
        for chunk in chunks:
            summary = await self._summarize_chunk(chunk, rag_contexts)
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # Generate final summary from combined summaries
        final_summary = await self._summarize_chunk(combined_summary, rag_contexts)
        
        return {
            'type': 'full_text',
            'sections': {'main': final_summary},
            'full_text': final_summary,
            'key_points': self._extract_key_points_from_text(final_summary)
        }
    
    def _generate_extractive_summary(self, text: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback extractive summarization when model loading fails."""
        sentences = text.split('. ')
        
        # Simple scoring based on keyword frequency
        word_freq = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            for word in words:
                score += word_freq.get(word, 0)
            
            if len(words) > 0:
                score = score / len(words)  # Normalize by length
            
            sentence_scores.append((score, i, sentence))
        
        # Select top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = sentence_scores[:5]  # Top 5 sentences
        top_sentences.sort(key=lambda x: x[1])  # Sort by original order
        
        summary = '. '.join([sent[2] for sent in top_sentences])
        
        return {
            'type': 'extractive',
            'sections': {'main': summary},
            'full_text': summary,
            'key_points': [sent[2] for sent in top_sentences[:3]]
        }
    
    async def _summarize_chunk(self, text: str, rag_contexts: Optional[List[str]] = None) -> str:
        """Summarize a text chunk with optional RAG context."""
        if rag_contexts:
            context_text = '\n\n'.join(rag_contexts)
            prompt = (
                "You are an expert research assistant. Using the MAIN PAPER below, write a summary as if explaining it to a fellow researcher. "
                "You may refer to the CONTEXT from similar papers for background or comparison, but the summary must be about the MAIN PAPER.\n\n"
                "CONTEXT (supporting, from similar papers):\n" + context_text +
                "\n\nMAIN PAPER:\n" + text +
                "\n\nSummary:"
            )
        else:
            prompt = f"Summarize the following academic text concisely:\n\n{text}\n\nSummary:"
        self.logger.info(f"Prompt sent to model (first 500 chars): {prompt[:500]}")
        if self.model == "fallback":
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) if len(sentences) >= 3 else text
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        device = self.model.device if hasattr(self.model, 'device') else self.device
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        summary = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return summary.strip()
    
    def _create_section_prompt(self, text: str, section_type: str) -> str:
        """Create appropriate prompt for different section types."""
        prompts = {
            'abstract': f"Summarize this abstract concisely:\n\n{text}\n\nSummary:",
            'introduction': f"Summarize the key points from this introduction:\n\n{text}\n\nKey points:",
            'methodology': f"Summarize the methodology described:\n\n{text}\n\nMethodology summary:",
            'results': f"Summarize the main results and findings:\n\n{text}\n\nResults summary:",
            'conclusion': f"Summarize the conclusions and implications:\n\n{text}\n\nConclusions:"
        }
        
        return prompts.get(section_type, f"Summarize this {section_type} section:\n\n{text}\n\nSummary:")
    
    def _extract_section(self, text: str, section_type: str) -> Optional[str]:
        """Extract text from a specific section."""
        text_lower = text.lower()
        
        # Find section start
        section_keywords = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', '1. introduction'],
            'methodology': ['methodology', 'methods', 'approach'],
            'results': ['results', 'experiments', 'evaluation'],
            'conclusion': ['conclusion', 'conclusions']
        }
        
        keywords = section_keywords.get(section_type, [section_type])
        
        for keyword in keywords:
            start_idx = text_lower.find(keyword)
            if start_idx != -1:
                # Find next section or end of text
                end_idx = len(text)
                for other_type, other_keywords in section_keywords.items():
                    if other_type != section_type:
                        for other_keyword in other_keywords:
                            other_idx = text_lower.find(other_keyword, start_idx + len(keyword))
                            if other_idx != -1 and other_idx < end_idx:
                                end_idx = other_idx
                
                return text[start_idx:end_idx].strip()
        
        return None
    
    def _extract_section_by_position(self, text: str, section_info: Dict[str, Any]) -> Optional[str]:
        """Extract section text based on position information."""
        start_pos = section_info.get('start_position', 0)
        
        # Find end position (next section or end of text)
        end_pos = len(text)
        
        # Extract text
        section_text = text[start_pos:end_pos]
        
        # Clean up - take first reasonable chunk
        lines = section_text.split('\n')
        relevant_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Skip very short lines
                relevant_lines.append(line)
            
            # Stop at next section header
            if len(relevant_lines) > 0 and (
                line.lower().startswith('abstract') or
                line.lower().startswith('introduction') or
                line.lower().startswith('methodology') or
                line.lower().startswith('results') or
                line.lower().startswith('conclusion')
            ):
                break
        
        return '\n'.join(relevant_lines[:20])  # Limit to first 20 relevant lines
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """Split text into manageable chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _combine_section_summaries(self, sections: Dict[str, str]) -> str:
        """Combine section summaries into a coherent full summary."""
        summary_parts = []
        
        # Order sections logically
        section_order = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
        
        for section_type in section_order:
            if section_type in sections:
                summary_parts.append(f"{section_type.title()}: {sections[section_type]}")
        
        # Add any remaining sections
        for section_type, summary in sections.items():
            if section_type not in section_order:
                summary_parts.append(f"{section_type.title()}: {summary}")
        
        return '\n\n'.join(summary_parts)
    
    def _extract_key_points(self, sections: Dict[str, str]) -> List[str]:
        """Extract key points from section summaries."""
        key_points = []
        
        for section_type, summary in sections.items():
            # Extract first sentence as key point
            sentences = summary.split('. ')
            if sentences:
                key_points.append(f"{section_type.title()}: {sentences[0]}")
        
        return key_points
    
    def _extract_key_points_from_text(self, text: str) -> List[str]:
        """Extract key points from summary text."""
        sentences = text.split('. ')
        # Return first 3 sentences as key points
        return sentences[:3] if len(sentences) >= 3 else sentences
