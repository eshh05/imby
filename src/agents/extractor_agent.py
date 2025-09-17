"""
Extractor Agent - Handles PDF parsing and content extraction.
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import re
import logging

class ExtractorAgent(BaseAgent):
    """Agent responsible for extracting content from PDF files."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("extractor", config)
        self.extraction_method = config.get('method', 'pdfplumber') if config else 'pdfplumber'
        
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Extract text and metadata from PDF file.
        
        Args:
            input_data: Dict containing 'file_path' or 'file_content'
            
        Returns:
            AgentResult with extracted content
        """
        try:
            if 'file_path' in input_data:
                content = self._extract_from_file(input_data['file_path'])
            elif 'file_content' in input_data:
                content = self._extract_from_bytes(input_data['file_content'])
            else:
                return AgentResult(
                    success=False,
                    data={},
                    error="Missing required 'file_path' or 'file_content' field"
                )
            
            result = AgentResult(
                success=True,
                data={
                    'text': content['text'],
                    'metadata': content['metadata'],
                    'figures': content.get('figures', []),
                    'tables': content.get('tables', [])
                },
                metadata={
                    'extraction_method': self.extraction_method,
                    'page_count': content['metadata'].get('page_count', 0),
                    'text_length': len(content['text'])
                }
            )
            
            self.log_execution(input_data, result)
            return result
            
        except Exception as e:
            error_msg = f"Error in extraction: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data={},
                error=error_msg
            )
    
    def _extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract content from PDF file path."""
        if self.extraction_method == 'pdfplumber':
            return self._extract_with_pdfplumber(file_path)
        elif self.extraction_method == 'pymupdf':
            return self._extract_with_pymupdf(file_path)
        else:
            return self._extract_with_pypdf2(file_path)
    
    def _extract_from_bytes(self, file_content: bytes) -> Dict[str, Any]:
        """Extract content from PDF bytes."""
        # For bytes, we'll use PyMuPDF as it handles bytes well
        doc = fitz.open(stream=file_content, filetype="pdf")
        return self._process_pymupdf_doc(doc)
    
    def _extract_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract using pdfplumber (best for tables and layout)."""
        content = {
            'text': '',
            'metadata': {},
            'figures': [],
            'tables': []
        }
        
        with pdfplumber.open(file_path) as pdf:
            content['metadata'] = {
                'page_count': len(pdf.pages),
                'title': pdf.metadata.get('Title', ''),
                'author': pdf.metadata.get('Author', ''),
                'creation_date': pdf.metadata.get('CreationDate', '')
            }
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    content['text'] += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    content['tables'].append({
                        'page': page_num + 1,
                        'data': table
                    })
        
        content['text'] = self._clean_text(content['text'])
        return content
    
    def _extract_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """Extract using PyMuPDF (good for text and images)."""
        doc = fitz.open(file_path)
        return self._process_pymupdf_doc(doc)
    
    def _process_pymupdf_doc(self, doc) -> Dict[str, Any]:
        """Process PyMuPDF document object."""
        content = {
            'text': '',
            'metadata': {},
            'figures': [],
            'tables': []
        }
        
        # Extract metadata
        content['metadata'] = {
            'page_count': doc.page_count,
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'creation_date': doc.metadata.get('creationDate', '')
        }
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text
            page_text = page.get_text()
            content['text'] += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            # Extract images/figures
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                content['figures'].append({
                    'page': page_num + 1,
                    'index': img_index,
                    'xref': img[0]
                })
        
        doc.close()
        content['text'] = self._clean_text(content['text'])
        return content
    
    def _extract_with_pypdf2(self, file_path: str) -> Dict[str, Any]:
        """Extract using PyPDF2 (basic text extraction)."""
        content = {
            'text': '',
            'metadata': {},
            'figures': [],
            'tables': []
        }
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            if pdf_reader.metadata:
                content['metadata'] = {
                    'page_count': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', '')
                }
            
            # Extract text
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                content['text'] += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        content['text'] = self._clean_text(content['text'])
        return content
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be headers/footers
            if len(line) > 3 and not re.match(r'^\d+$', line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from text using regex patterns."""
        citations = []
        
        # Pattern for numbered citations [1], [2], etc.
        numbered_pattern = r'\[(\d+)\]'
        numbered_matches = re.findall(numbered_pattern, text)
        
        for match in numbered_matches:
            citations.append({
                'type': 'numbered',
                'number': int(match),
                'text': f'[{match}]'
            })
        
        # Pattern for author-year citations (Smith, 2020)
        author_year_pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)'
        author_year_matches = re.findall(author_year_pattern, text)
        
        for author, year in author_year_matches:
            citations.append({
                'type': 'author_year',
                'author': author,
                'year': int(year),
                'text': f'({author}, {year})'
            })
        
        return citations
