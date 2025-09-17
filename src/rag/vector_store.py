"""
Vector store implementation for RAG (Retrieval-Augmented Generation) integration.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import json
from dataclasses import dataclass
import requests
import tempfile
import mimetypes
from urllib.parse import urlparse
import pdfplumber
from bs4 import BeautifulSoup

@dataclass
class Document:
    """Represents a document in the vector store."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class PaperVectorStore:
    """Vector store for research papers using ChromaDB."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.collection_name = self.config.get('collection_name', 'research_papers')
        self.persist_directory = self.config.get('persist_directory', 'data/vector_store')
        self.embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        
        self.logger = logging.getLogger("vector_store")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB
        self._init_chromadb()
        
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Research papers for academic summarization"}
            )
            
            self.logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_paper(self, paper_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Add a research paper to the vector store."""
        try:
            # Convert any list values in metadata to strings
            for k, v in metadata.items():
                if isinstance(v, list):
                    metadata[k] = ", ".join(map(str, v))

            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Add to collection
            self.collection.add(
                ids=[paper_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            
            self.logger.info(f"Added paper {paper_id} to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add paper {paper_id}: {str(e)}")
            return False
    
    def add_papers_batch(self, papers: List[Dict[str, Any]]) -> int:
        """Add multiple papers to the vector store in batch, overwriting existing IDs."""
        try:
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            added_count = 0
            for paper in papers:
                paper_id = paper['id']
                content = paper.get('content')
                metadata = paper.get('metadata', {})
                # If metadata is empty, populate from top-level fields
                if not metadata:
                    metadata = {k: v for k, v in paper.items() if k not in ['id', 'content', 'url']}
                # Convert any list values in metadata to strings
                for k, v in metadata.items():
                    if isinstance(v, list):
                        metadata[k] = ", ".join(map(str, v))
                # If content is missing but url is present, fetch and extract
                if not content and paper.get('url'):
                    self.logger.info(f"Fetching and extracting content for paper {paper_id} from {paper['url']}")
                    content = self._fetch_and_extract_content(paper['url'])
                if not content:
                    self.logger.error(f"Skipping paper {paper_id}: no content and could not extract from url")
                    continue
                # Overwrite: delete existing paper with this ID
                try:
                    self.collection.delete(ids=[paper_id])
                except Exception as e:
                    self.logger.warning(f"Failed to delete existing paper {paper_id} before overwrite: {e}")
                # Generate embedding
                embedding = self.embedding_model.encode(content).tolist()
                ids.append(paper_id)
                documents.append(content)
                metadatas.append(metadata)
                embeddings.append(embedding)
                added_count += 1
            if ids:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                self.logger.info(f"Added {added_count} papers to vector store (with overwrite)")
            return added_count
        except Exception as e:
            self.logger.error(f"Failed to add papers batch: {str(e)}")
            return 0
    
    def _download_file(self, url: str) -> Optional[str]:
        """Download file from URL to a temporary location and return the path."""
        try:
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()
            parsed = urlparse(url)
            ext = mimetypes.guess_extension(response.headers.get('content-type', '')) or '.pdf'
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                return tmp_file.name
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return None

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            return text.strip()
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return None

    def _extract_text_from_html(self, html_path: str) -> Optional[str]:
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            self.logger.error(f"Failed to extract text from HTML {html_path}: {e}")
            return None

    def _fetch_and_extract_content(self, url: str) -> Optional[str]:
        """Download a file from URL and extract its text."""
        file_path = self._download_file(url)
        if not file_path:
            return None
        if file_path.endswith('.pdf'):
            return self._extract_text_from_pdf(file_path)
        elif file_path.endswith('.html') or file_path.endswith('.htm'):
            return self._extract_text_from_html(file_path)
        else:
            # Try PDF first, fallback to HTML
            text = self._extract_text_from_pdf(file_path)
            if text:
                return text
            return self._extract_text_from_html(file_path)

    def search_similar_papers(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for papers similar to the query."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            self.logger.info(f"Found {len(formatted_results)} similar papers for query")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search papers: {str(e)}")
            return []
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], n_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers by metadata filters."""
        try:
            # Build where clause for ChromaDB
            where_clause = {}
            for key, value in metadata_filter.items():
                where_clause[key] = value
            
            results = self.collection.get(
                where=where_clause,
                limit=n_results,
                include=['documents', 'metadatas']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'])):
                formatted_results.append({
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            self.logger.info(f"Found {len(formatted_results)} papers matching metadata filter")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search by metadata: {str(e)}")
            return []
    
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific paper by ID."""
        try:
            results = self.collection.get(
                ids=[paper_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get paper {paper_id}: {str(e)}")
            return None
    
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from the vector store."""
        try:
            self.collection.delete(ids=[paper_id])
            self.logger.info(f"Deleted paper {paper_id} from vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete paper {paper_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            return {
                'total_papers': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def create_paper_chunks(self, paper_content: str, paper_metadata: Dict[str, Any], 
                           chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split a paper into chunks for better retrieval."""
        
        # Split content into sentences first
        sentences = paper_content.split('. ')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, create a new chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        **paper_metadata,
                        'chunk_index': len(chunks),
                        'chunk_length': len(chunk_text)
                    }
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-overlap//100:] if overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                'content': chunk_text,
                'metadata': {
                    **paper_metadata,
                    'chunk_index': len(chunks),
                    'chunk_length': len(chunk_text)
                }
            })
        
        return chunks
