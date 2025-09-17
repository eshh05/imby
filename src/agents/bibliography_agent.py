"""
Bibliography Agent - Citation extraction and formatting.
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult
import re
from dataclasses import dataclass
import logging

@dataclass
class Citation:
    """Represents a citation found in the paper."""
    text: str
    citation_type: str  # 'numbered', 'author_year', 'doi'
    authors: List[str] = None
    title: str = None
    year: int = None
    journal: str = None
    doi: str = None
    url: str = None

class BibliographyAgent(BaseAgent):
    """Agent responsible for extracting and formatting citations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("bibliography", config)
        self.citation_styles = ['apa', 'mla', 'ieee', 'chicago']
        
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Extract citations and generate bibliography.
        
        Args:
            input_data: Dict containing 'text' and optional 'citation_plan'
            
        Returns:
            AgentResult with extracted citations and formatted bibliography
        """
        try:
            if not self.validate_input(input_data, ['text']):
                return AgentResult(
                    success=False,
                    data={},
                    error="Missing required 'text' field"
                )
            
            text = input_data['text']
            citation_plan = input_data.get('citation_plan', {})
            
            # Extract citations
            citations = self._extract_citations(text, citation_plan)
            
            # Generate bibliography in multiple formats
            bibliographies = {}
            for style in self.citation_styles:
                bibliographies[style] = self._format_bibliography(citations, style)
            
            result = AgentResult(
                success=True,
                data={
                    'citations': [self._citation_to_dict(c) for c in citations],
                    'bibliographies': bibliographies,
                    'citation_count': len(citations),
                    'reference_section': self._extract_reference_section(text)
                },
                metadata={
                    'citation_style_detected': citation_plan.get('citation_style', 'unknown'),
                    'has_reference_section': citation_plan.get('has_reference_section', False),
                    'extraction_method': citation_plan.get('extraction_method', 'regex_pattern_matching')
                }
            )
            
            self.log_execution(input_data, result)
            return result
            
        except Exception as e:
            error_msg = f"Error in bibliography processing: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data={},
                error=error_msg
            )
    
    def _extract_citations(self, text: str, citation_plan: Dict[str, Any]) -> List[Citation]:
        """Extract citations from text based on detected style."""
        citations = []
        
        citation_style = citation_plan.get('citation_style', 'unknown')
        
        if citation_style == 'numbered':
            citations.extend(self._extract_numbered_citations(text))
        elif citation_style == 'author_year':
            citations.extend(self._extract_author_year_citations(text))
        else:
            # Try both methods
            citations.extend(self._extract_numbered_citations(text))
            citations.extend(self._extract_author_year_citations(text))
        
        # Extract DOI citations
        citations.extend(self._extract_doi_citations(text))
        
        # Remove duplicates
        unique_citations = []
        seen_texts = set()
        
        for citation in citations:
            if citation.text not in seen_texts:
                unique_citations.append(citation)
                seen_texts.add(citation.text)
        
        return unique_citations
    
    def _extract_numbered_citations(self, text: str) -> List[Citation]:
        """Extract numbered citations like [1], [2], etc."""
        citations = []
        
        # Pattern for numbered citations
        pattern = r'\[(\d+)\]'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            number = int(match.group(1))
            citations.append(Citation(
                text=match.group(0),
                citation_type='numbered',
                authors=[],
                year=None
            ))
        
        return citations
    
    def _extract_author_year_citations(self, text: str) -> List[Citation]:
        """Extract author-year citations like (Smith, 2020)."""
        citations = []
        
        # Pattern for author-year citations
        patterns = [
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)',  # (Smith, 2020)
            r'([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+\((\d{4})\)',    # Smith (2020)
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+(\d{4})\)',    # (Smith 2020)
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                author = match.group(1)
                year = int(match.group(2))
                
                citations.append(Citation(
                    text=match.group(0),
                    citation_type='author_year',
                    authors=[author],
                    year=year
                ))
        
        return citations
    
    def _extract_doi_citations(self, text: str) -> List[Citation]:
        """Extract DOI citations."""
        citations = []
        
        # Pattern for DOI
        doi_pattern = r'doi:\s*(10\.\d+/[^\s]+)'
        matches = re.finditer(doi_pattern, text, re.IGNORECASE)
        
        for match in matches:
            doi = match.group(1)
            citations.append(Citation(
                text=match.group(0),
                citation_type='doi',
                doi=doi
            ))
        
        return citations
    
    def _extract_reference_section(self, text: str) -> Optional[str]:
        """Extract the references/bibliography section from the paper."""
        text_lower = text.lower()
        
        # Find references section
        ref_keywords = ['references', 'bibliography', 'works cited']
        
        for keyword in ref_keywords:
            start_idx = text_lower.find(keyword)
            if start_idx != -1:
                # Take everything from references to end (or next major section)
                ref_section = text[start_idx:]
                
                # Try to find end of references (acknowledgments, appendix, etc.)
                end_keywords = ['acknowledgment', 'appendix', 'author information']
                end_idx = len(ref_section)
                
                for end_keyword in end_keywords:
                    end_pos = ref_section.lower().find(end_keyword)
                    if end_pos != -1 and end_pos < end_idx:
                        end_idx = end_pos
                
                return ref_section[:end_idx].strip()
        
        return None
    
    def _format_bibliography(self, citations: List[Citation], style: str) -> List[str]:
        """Format citations in the specified style."""
        formatted_citations = []
        
        for citation in citations:
            if style == 'apa':
                formatted = self._format_apa(citation)
            elif style == 'mla':
                formatted = self._format_mla(citation)
            elif style == 'ieee':
                formatted = self._format_ieee(citation)
            elif style == 'chicago':
                formatted = self._format_chicago(citation)
            else:
                formatted = citation.text
            
            if formatted:
                formatted_citations.append(formatted)
        
        return formatted_citations
    
    def _format_apa(self, citation: Citation) -> str:
        """Format citation in APA style."""
        if citation.authors and citation.year:
            authors_str = ', '.join(citation.authors)
            if citation.title and citation.journal:
                return f"{authors_str} ({citation.year}). {citation.title}. {citation.journal}."
            else:
                return f"{authors_str} ({citation.year}). [Citation details incomplete]"
        elif citation.doi:
            return f"DOI: {citation.doi}"
        else:
            return f"[{citation.citation_type}] {citation.text}"
    
    def _format_mla(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        if citation.authors and citation.year:
            authors_str = ', '.join(citation.authors)
            if citation.title and citation.journal:
                return f"{authors_str}. \"{citation.title}.\" {citation.journal}, {citation.year}."
            else:
                return f"{authors_str}. {citation.year}. [Citation details incomplete]"
        elif citation.doi:
            return f"DOI: {citation.doi}"
        else:
            return f"[{citation.citation_type}] {citation.text}"
    
    def _format_ieee(self, citation: Citation) -> str:
        """Format citation in IEEE style."""
        if citation.authors and citation.year:
            authors_str = ', '.join(citation.authors)
            if citation.title and citation.journal:
                return f"{authors_str}, \"{citation.title},\" {citation.journal}, {citation.year}."
            else:
                return f"{authors_str}, {citation.year}. [Citation details incomplete]"
        elif citation.doi:
            return f"DOI: {citation.doi}"
        else:
            return f"[{citation.citation_type}] {citation.text}"
    
    def _format_chicago(self, citation: Citation) -> str:
        """Format citation in Chicago style."""
        if citation.authors and citation.year:
            authors_str = ', '.join(citation.authors)
            if citation.title and citation.journal:
                return f"{authors_str}. \"{citation.title}.\" {citation.journal} ({citation.year})."
            else:
                return f"{authors_str}. {citation.year}. [Citation details incomplete]"
        elif citation.doi:
            return f"DOI: {citation.doi}"
        else:
            return f"[{citation.citation_type}] {citation.text}"
    
    def _citation_to_dict(self, citation: Citation) -> Dict[str, Any]:
        """Convert Citation object to dictionary."""
        return {
            'text': citation.text,
            'type': citation.citation_type,
            'authors': citation.authors or [],
            'title': citation.title,
            'year': citation.year,
            'journal': citation.journal,
            'doi': citation.doi,
            'url': citation.url
        }
