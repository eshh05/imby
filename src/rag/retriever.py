"""
RAG Retriever for contextual paper retrieval during summarization.
"""

from typing import Dict, Any, List, Optional
import logging
from .vector_store import PaperVectorStore

class RAGRetriever:
    """Retrieval-Augmented Generation component for paper summarization."""
    
    def __init__(self, vector_store: PaperVectorStore, config: Dict[str, Any] = None):
        self.vector_store = vector_store
        self.config = config or {}
        self.logger = logging.getLogger("rag_retriever")
        
        # Configuration
        self.max_retrieved_papers = self.config.get('max_retrieved_papers', 3)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.context_window_size = self.config.get('context_window_size', 2000)
        
    def retrieve_context(self, query_text: str, paper_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve relevant context for paper summarization.
        
        Args:
            query_text: Text to find similar content for
            paper_metadata: Optional metadata filters
            
        Returns:
            Dictionary containing retrieved context and metadata
        """
        try:
            # Search for similar papers
            similar_papers = self.vector_store.search_similar_papers(
                query=query_text,
                n_results=self.max_retrieved_papers * 2  # Get more to filter by threshold
            )
            
            # Filter by similarity threshold
            relevant_papers = [
                paper for paper in similar_papers 
                if paper['similarity_score'] >= self.similarity_threshold
            ][:self.max_retrieved_papers]
            
            # Apply metadata filters if provided
            if paper_metadata:
                relevant_papers = self._filter_by_metadata(relevant_papers, paper_metadata)
            
            # Extract context
            context = self._extract_context(relevant_papers, query_text)
            
            return {
                'context': context,
                'retrieved_papers': relevant_papers,
                'retrieval_metadata': {
                    'query_length': len(query_text),
                    'papers_found': len(similar_papers),
                    'papers_relevant': len(relevant_papers),
                    'context_length': len(context)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in context retrieval: {str(e)}")
            return {
                'context': '',
                'retrieved_papers': [],
                'retrieval_metadata': {'error': str(e)}
            }
    
    def retrieve_similar_sections(self, section_text: str, section_type: str) -> List[Dict[str, Any]]:
        """Retrieve similar sections from other papers."""
        try:
            # Create section-specific query
            query = f"{section_type}: {section_text[:500]}"  # Limit query length
            
            # Search with metadata filter for section type
            similar_papers = self.vector_store.search_similar_papers(
                query=query,
                n_results=self.max_retrieved_papers
            )
            
            # Filter for papers with similar section types in metadata
            relevant_sections = []
            for paper in similar_papers:
                if paper['similarity_score'] >= self.similarity_threshold:
                    # Extract relevant section if available
                    section_content = self._extract_section_from_paper(
                        paper['content'], section_type
                    )
                    if section_content:
                        relevant_sections.append({
                            'paper_id': paper['id'],
                            'section_type': section_type,
                            'content': section_content,
                            'similarity_score': paper['similarity_score'],
                            'metadata': paper['metadata']
                        })
            
            return relevant_sections
            
        except Exception as e:
            self.logger.error(f"Error retrieving similar sections: {str(e)}")
            return []
    
    def retrieve_citation_context(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrieve context for citations to enhance bibliography."""
        try:
            citation_contexts = {}
            
            for citation in citations:
                if citation['type'] == 'author_year' and citation.get('authors'):
                    # Search for papers by the same authors
                    author_query = ' '.join(citation['authors'])
                    
                    similar_papers = self.vector_store.search_by_metadata(
                        metadata_filter={'author': author_query},
                        n_results=3
                    )
                    
                    if not similar_papers:
                        # Fallback to content search
                        similar_papers = self.vector_store.search_similar_papers(
                            query=author_query,
                            n_results=3
                        )
                    
                    citation_contexts[citation['text']] = similar_papers
            
            return citation_contexts
            
        except Exception as e:
            self.logger.error(f"Error retrieving citation context: {str(e)}")
            return {}
    
    def _filter_by_metadata(self, papers: List[Dict[str, Any]], 
                           metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter papers by metadata criteria."""
        filtered_papers = []
        
        for paper in papers:
            paper_metadata = paper.get('metadata', {})
            
            # Check if paper matches all filter criteria
            matches = True
            for key, value in metadata_filter.items():
                if key in paper_metadata:
                    if isinstance(value, str):
                        # Case-insensitive string matching
                        if value.lower() not in str(paper_metadata[key]).lower():
                            matches = False
                            break
                    else:
                        if paper_metadata[key] != value:
                            matches = False
                            break
            
            if matches:
                filtered_papers.append(paper)
        
        return filtered_papers
    
    def _extract_context(self, papers: List[Dict[str, Any]], query_text: str) -> str:
        """Extract and combine context from retrieved papers."""
        context_parts = []
        current_length = 0
        
        for paper in papers:
            paper_content = paper['content']
            paper_id = paper['id']
            
            # Find most relevant excerpt from the paper
            excerpt = self._find_relevant_excerpt(paper_content, query_text)
            
            # Add paper context with attribution
            paper_context = f"[Paper {paper_id}]: {excerpt}"
            
            # Check if adding this context would exceed window size
            if current_length + len(paper_context) > self.context_window_size:
                break
            
            context_parts.append(paper_context)
            current_length += len(paper_context)
        
        return '\n\n'.join(context_parts)
    
    def _find_relevant_excerpt(self, paper_content: str, query_text: str, 
                              excerpt_length: int = 500) -> str:
        """Find the most relevant excerpt from a paper."""
        
        # Split into sentences
        sentences = paper_content.split('. ')
        
        # Simple relevance scoring based on word overlap
        query_words = set(query_text.lower().split())
        
        best_excerpt = ""
        best_score = 0
        
        # Sliding window approach
        for i in range(len(sentences)):
            # Take a window of sentences
            window_end = min(i + 3, len(sentences))  # 3-sentence window
            window_text = '. '.join(sentences[i:window_end])
            
            if len(window_text) > excerpt_length:
                window_text = window_text[:excerpt_length] + "..."
            
            # Score based on word overlap
            window_words = set(window_text.lower().split())
            overlap = len(query_words.intersection(window_words))
            score = overlap / len(query_words) if query_words else 0
            
            if score > best_score:
                best_score = score
                best_excerpt = window_text
        
        return best_excerpt if best_excerpt else paper_content[:excerpt_length] + "..."
    
    def _extract_section_from_paper(self, paper_content: str, section_type: str) -> Optional[str]:
        """Extract a specific section from paper content."""
        
        section_keywords = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', '1. introduction'],
            'methodology': ['methodology', 'methods', 'approach'],
            'results': ['results', 'experiments', 'evaluation'],
            'conclusion': ['conclusion', 'conclusions']
        }
        
        keywords = section_keywords.get(section_type.lower(), [section_type.lower()])
        content_lower = paper_content.lower()
        
        for keyword in keywords:
            start_idx = content_lower.find(keyword)
            if start_idx != -1:
                # Find end of section (next section or reasonable cutoff)
                end_idx = start_idx + 1000  # Default cutoff
                
                # Look for next section
                for other_section, other_keywords in section_keywords.items():
                    if other_section != section_type.lower():
                        for other_keyword in other_keywords:
                            other_idx = content_lower.find(other_keyword, start_idx + len(keyword))
                            if other_idx != -1 and other_idx < end_idx:
                                end_idx = other_idx
                
                section_text = paper_content[start_idx:end_idx].strip()
                return section_text[:800] + "..." if len(section_text) > 800 else section_text
        
        return None
