"""
Evaluation metrics for research paper summarization quality.
"""

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging
from typing import Dict, Any, List, Optional, Tuple
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SummarizationEvaluator:
    """Comprehensive evaluation metrics for academic paper summarization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("evaluator")
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Initialize semantic similarity model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def evaluate_summary(self, generated_summary: str, reference_summary: str, 
                        source_text: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a generated summary.
        
        Args:
            generated_summary: AI-generated summary
            reference_summary: Human-written reference summary
            source_text: Original paper text (optional)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        
        metrics = {}
        
        # ROUGE Scores
        metrics['rouge'] = self._calculate_rouge_scores(generated_summary, reference_summary)
        
        # Semantic Similarity
        metrics['semantic_similarity'] = self._calculate_semantic_similarity(
            generated_summary, reference_summary
        )
        
        # Content Quality Metrics
        metrics['content_quality'] = self._evaluate_content_quality(
            generated_summary, reference_summary
        )
        
        # Readability Metrics
        metrics['readability'] = self._calculate_readability_metrics(generated_summary)
        
        # Coverage Metrics (if source text provided)
        if source_text:
            metrics['coverage'] = self._calculate_coverage_metrics(
                generated_summary, source_text
            )
        
        # Overall Score
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def evaluate_citation_extraction(self, extracted_citations: List[Dict[str, Any]], 
                                   ground_truth_citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate citation extraction accuracy."""
        
        if not ground_truth_citations:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0
            }
        
        # Convert to sets for comparison
        extracted_set = set()
        for citation in extracted_citations:
            if citation.get('authors') and citation.get('year'):
                key = f"{citation['authors'][0]}_{citation['year']}"
                extracted_set.add(key)
            elif citation.get('text'):
                extracted_set.add(citation['text'])
        
        ground_truth_set = set()
        for citation in ground_truth_citations:
            if citation.get('authors') and citation.get('year'):
                key = f"{citation['authors'][0]}_{citation['year']}"
                ground_truth_set.add(key)
            elif citation.get('text'):
                ground_truth_set.add(citation['text'])
        
        # Calculate metrics
        true_positives = len(extracted_set.intersection(ground_truth_set))
        false_positives = len(extracted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - extracted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def _calculate_semantic_similarity(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate semantic similarity using sentence transformers."""
        
        # Generate embeddings
        generated_embedding = self.semantic_model.encode([generated])
        reference_embedding = self.semantic_model.encode([reference])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(generated_embedding, reference_embedding)[0][0]
        
        # Sentence-level similarity
        generated_sentences = sent_tokenize(generated)
        reference_sentences = sent_tokenize(reference)
        
        if generated_sentences and reference_sentences:
            gen_sent_embeddings = self.semantic_model.encode(generated_sentences)
            ref_sent_embeddings = self.semantic_model.encode(reference_sentences)
            
            # Find best matches for each generated sentence
            sentence_similarities = []
            for gen_emb in gen_sent_embeddings:
                max_sim = max(cosine_similarity([gen_emb], ref_sent_embeddings)[0])
                sentence_similarities.append(max_sim)
            
            avg_sentence_similarity = np.mean(sentence_similarities)
        else:
            avg_sentence_similarity = 0.0
        
        return {
            'overall_similarity': float(similarity),
            'sentence_similarity': float(avg_sentence_similarity)
        }
    
    def _evaluate_content_quality(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate content quality metrics."""
        
        # Length ratio
        gen_words = len(word_tokenize(generated))
        ref_words = len(word_tokenize(reference))
        length_ratio = gen_words / ref_words if ref_words > 0 else 0
        
        # Vocabulary overlap
        gen_vocab = set(word.lower() for word in word_tokenize(generated) 
                       if word.isalpha() and word.lower() not in self.stop_words)
        ref_vocab = set(word.lower() for word in word_tokenize(reference) 
                       if word.isalpha() and word.lower() not in self.stop_words)
        
        vocab_overlap = len(gen_vocab.intersection(ref_vocab)) / len(ref_vocab.union(gen_vocab)) if ref_vocab.union(gen_vocab) else 0
        
        # Key term preservation
        key_terms = self._extract_key_terms(reference)
        preserved_terms = sum(1 for term in key_terms if term.lower() in generated.lower())
        key_term_preservation = preserved_terms / len(key_terms) if key_terms else 0
        
        return {
            'length_ratio': length_ratio,
            'vocabulary_overlap': vocab_overlap,
            'key_term_preservation': key_term_preservation
        }
    
    def _calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return {
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0,
                'sentence_count': 0,
                'word_count': 0
            }
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        word_lengths = [len(word) for word in words if word.isalpha()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'sentence_count': len(sentences),
            'word_count': len(words)
        }
    
    def _calculate_coverage_metrics(self, summary: str, source_text: str) -> Dict[str, float]:
        """Calculate how well the summary covers the source text."""
        
        # Extract key concepts from source
        source_concepts = self._extract_key_terms(source_text)
        summary_concepts = self._extract_key_terms(summary)
        
        # Coverage of key concepts
        covered_concepts = sum(1 for concept in source_concepts 
                             if any(concept.lower() in summary.lower() for summary in [summary]))
        concept_coverage = covered_concepts / len(source_concepts) if source_concepts else 0
        
        # Section coverage (if sections are identifiable)
        source_sections = self._identify_sections(source_text)
        summary_sections = self._identify_sections(summary)
        
        section_coverage = len(set(source_sections).intersection(set(summary_sections))) / len(source_sections) if source_sections else 0
        
        return {
            'concept_coverage': concept_coverage,
            'section_coverage': section_coverage,
            'compression_ratio': len(summary) / len(source_text) if source_text else 0
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text using simple frequency analysis."""
        
        words = word_tokenize(text.lower())
        
        # Filter words
        filtered_words = [
            word for word in words 
            if word.isalpha() and len(word) > 3 and word not in self.stop_words
        ]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top frequent words as key terms
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:20]]  # Top 20 terms
    
    def _identify_sections(self, text: str) -> List[str]:
        """Identify sections mentioned in text."""
        
        section_patterns = [
            'abstract', 'introduction', 'methodology', 'methods',
            'results', 'discussion', 'conclusion', 'references'
        ]
        
        found_sections = []
        text_lower = text.lower()
        
        for section in section_patterns:
            if section in text_lower:
                found_sections.append(section)
        
        return found_sections
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall score."""
        
        weights = {
            'rouge': 0.3,
            'semantic_similarity': 0.25,
            'content_quality': 0.25,
            'readability': 0.1,
            'coverage': 0.1
        }
        
        score = 0.0
        
        # ROUGE score (average F1)
        if 'rouge' in metrics:
            rouge_avg = (metrics['rouge']['rouge1_f'] + 
                        metrics['rouge']['rouge2_f'] + 
                        metrics['rouge']['rougeL_f']) / 3
            score += weights['rouge'] * rouge_avg
        
        # Semantic similarity
        if 'semantic_similarity' in metrics:
            sem_score = metrics['semantic_similarity']['overall_similarity']
            score += weights['semantic_similarity'] * sem_score
        
        # Content quality
        if 'content_quality' in metrics:
            content_score = (metrics['content_quality']['vocabulary_overlap'] + 
                           metrics['content_quality']['key_term_preservation']) / 2
            score += weights['content_quality'] * content_score
        
        # Readability (normalized)
        if 'readability' in metrics:
            # Simple readability score based on reasonable ranges
            sent_len = metrics['readability']['avg_sentence_length']
            word_len = metrics['readability']['avg_word_length']
            
            # Normalize to 0-1 (ideal ranges: 15-25 words/sentence, 4-6 chars/word)
            sent_score = max(0, 1 - abs(sent_len - 20) / 20) if sent_len > 0 else 0
            word_score = max(0, 1 - abs(word_len - 5) / 5) if word_len > 0 else 0
            
            readability_score = (sent_score + word_score) / 2
            score += weights['readability'] * readability_score
        
        # Coverage
        if 'coverage' in metrics:
            coverage_score = metrics['coverage']['concept_coverage']
            score += weights['coverage'] * coverage_score
        
        return min(1.0, max(0.0, score))  # Clamp to [0, 1]

class BatchEvaluator:
    """Evaluate multiple summaries in batch."""
    
    def __init__(self, evaluator: SummarizationEvaluator):
        self.evaluator = evaluator
        self.logger = logging.getLogger("batch_evaluator")
    
    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of summarization results."""
        
        individual_scores = []
        citation_scores = []
        
        for result in results:
            if not result.get('success', False):
                continue
            
            # Evaluate summary if reference is available
            if 'reference_summary' in result:
                summary_eval = self.evaluator.evaluate_summary(
                    generated_summary=result['summary']['full_text'],
                    reference_summary=result['reference_summary'],
                    source_text=result.get('source_text')
                )
                individual_scores.append(summary_eval)
            
            # Evaluate citations if ground truth is available
            if 'ground_truth_citations' in result:
                citation_eval = self.evaluator.evaluate_citation_extraction(
                    extracted_citations=result['bibliography']['citations'],
                    ground_truth_citations=result['ground_truth_citations']
                )
                citation_scores.append(citation_eval)
        
        # Aggregate scores
        aggregated = self._aggregate_scores(individual_scores, citation_scores)
        
        return {
            'individual_evaluations': individual_scores,
            'citation_evaluations': citation_scores,
            'aggregated_metrics': aggregated,
            'total_papers': len(results),
            'successful_papers': len([r for r in results if r.get('success', False)])
        }
    
    def _aggregate_scores(self, summary_scores: List[Dict], citation_scores: List[Dict]) -> Dict[str, Any]:
        """Aggregate individual scores into summary statistics."""
        
        aggregated = {}
        
        if summary_scores:
            # Aggregate ROUGE scores
            rouge_metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f']
            aggregated['rouge'] = {}
            
            for metric in rouge_metrics:
                values = [score['rouge'][metric] for score in summary_scores]
                aggregated['rouge'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            # Aggregate semantic similarity
            sem_values = [score['semantic_similarity']['overall_similarity'] for score in summary_scores]
            aggregated['semantic_similarity'] = {
                'mean': np.mean(sem_values),
                'std': np.std(sem_values),
                'min': np.min(sem_values),
                'max': np.max(sem_values)
            }
            
            # Aggregate overall scores
            overall_values = [score['overall_score'] for score in summary_scores]
            aggregated['overall_score'] = {
                'mean': np.mean(overall_values),
                'std': np.std(overall_values),
                'min': np.min(overall_values),
                'max': np.max(overall_values)
            }
        
        if citation_scores:
            # Aggregate citation metrics
            citation_metrics = ['precision', 'recall', 'f1_score']
            aggregated['citations'] = {}
            
            for metric in citation_metrics:
                values = [score[metric] for score in citation_scores]
                aggregated['citations'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated
