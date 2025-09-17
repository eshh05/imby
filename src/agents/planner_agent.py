"""
Planner Agent - Analyzes paper structure and creates processing strategy.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentResult
import logging

class PlannerAgent(BaseAgent):
    """Agent responsible for analyzing paper structure and planning processing strategy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("planner", config)
        
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze paper content and create processing plan.
        
        Args:
            input_data: Dict containing 'text' (extracted paper text)
            
        Returns:
            AgentResult with processing plan
        """
        try:
            if not self.validate_input(input_data, ['text']):
                return AgentResult(
                    success=False,
                    data={},
                    error="Missing required 'text' field"
                )
            
            text = input_data['text']
            plan = self._create_processing_plan(text)
            
            result = AgentResult(
                success=True,
                data={
                    'plan': plan,
                    'sections_identified': plan['sections'],
                    'processing_strategy': plan['strategy']
                },
                metadata={
                    'text_length': len(text),
                    'estimated_sections': len(plan['sections'])
                }
            )
            
            self.log_execution(input_data, result)
            return result
            
        except Exception as e:
            error_msg = f"Error in planner execution: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data={},
                error=error_msg
            )
    
    def _create_processing_plan(self, text: str) -> Dict[str, Any]:
        """Create a structured processing plan based on paper content."""
        
        # Identify paper sections
        sections = self._identify_sections(text)
        
        # Determine processing strategy
        strategy = {
            'summarization_approach': self._determine_summarization_approach(sections),
            'citation_extraction': self._plan_citation_extraction(text),
            'priority_sections': self._identify_priority_sections(sections)
        }
        
        return {
            'sections': sections,
            'strategy': strategy,
            'processing_order': self._determine_processing_order(sections)
        }
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify and locate paper sections."""
        sections = []
        
        # Common academic paper section patterns
        section_patterns = [
            ('abstract', ['abstract', 'summary']),
            ('introduction', ['introduction', '1. introduction', '1 introduction']),
            ('methodology', ['methodology', 'methods', 'approach', 'model']),
            ('results', ['results', 'experiments', 'evaluation', 'findings']),
            ('discussion', ['discussion', 'analysis', 'implications']),
            ('conclusion', ['conclusion', 'conclusions', 'summary']),
            ('references', ['references', 'bibliography', 'citations'])
        ]
        
        text_lower = text.lower()
        
        for section_type, keywords in section_patterns:
            for keyword in keywords:
                if keyword in text_lower:
                    # Find approximate position
                    start_pos = text_lower.find(keyword)
                    sections.append({
                        'type': section_type,
                        'keyword': keyword,
                        'start_position': start_pos,
                        'confidence': self._calculate_section_confidence(keyword, text_lower)
                    })
                    break
        
        # Sort by position in document
        sections.sort(key=lambda x: x['start_position'])
        return sections
    
    def _calculate_section_confidence(self, keyword: str, text: str) -> float:
        """Calculate confidence that a keyword represents a section header."""
        # Simple heuristic: check if keyword appears at line start or with numbers
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip().lower()
            if keyword in line_stripped:
                # Higher confidence if it's at the start of a line
                if line_stripped.startswith(keyword):
                    return 0.9
                # Medium confidence if it contains numbers (like "1. Introduction")
                elif any(char.isdigit() for char in line_stripped):
                    return 0.7
        return 0.5
    
    def _determine_summarization_approach(self, sections: List[Dict]) -> str:
        """Determine the best summarization approach based on identified sections."""
        section_types = [s['type'] for s in sections]
        
        if 'abstract' in section_types:
            return 'structured_with_abstract'
        elif len(section_types) >= 4:
            return 'structured_multi_section'
        else:
            return 'unstructured_full_text'
    
    def _plan_citation_extraction(self, text: str) -> Dict[str, Any]:
        """Plan citation extraction strategy."""
        # Check for common citation patterns
        has_numbered_refs = '[1]' in text or '(1)' in text
        has_author_year = '(' in text and ')' in text and any(year in text for year in ['2020', '2021', '2022', '2023', '2024'])
        
        return {
            'citation_style': 'numbered' if has_numbered_refs else 'author_year',
            'has_reference_section': 'references' in text.lower(),
            'extraction_method': 'regex_pattern_matching'
        }
    
    def _identify_priority_sections(self, sections: List[Dict]) -> List[str]:
        """Identify which sections should be prioritized for summarization."""
        priority_order = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
        identified_types = [s['type'] for s in sections]
        
        return [section_type for section_type in priority_order if section_type in identified_types]
    
    def _determine_processing_order(self, sections: List[Dict]) -> List[str]:
        """Determine the optimal order for processing sections."""
        # Process in document order but prioritize key sections
        return [s['type'] for s in sections]
