"""
Unit tests for the agent components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.agents.base_agent import BaseAgent, AgentResult
from src.agents.planner_agent import PlannerAgent
from src.agents.extractor_agent import ExtractorAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.bibliography_agent import BibliographyAgent

class TestBaseAgent:
    """Test the base agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        
        class TestAgent(BaseAgent):
            async def execute(self, input_data):
                return AgentResult(success=True, data={'test': 'data'})
        
        agent = TestAgent("test_agent", {"config_key": "config_value"})
        
        assert agent.name == "test_agent"
        assert agent.config == {"config_key": "config_value"}
    
    def test_validate_input(self):
        """Test input validation."""
        
        class TestAgent(BaseAgent):
            async def execute(self, input_data):
                return AgentResult(success=True, data={})
        
        agent = TestAgent("test_agent")
        
        # Valid input
        assert agent.validate_input({"key1": "value1", "key2": "value2"}, ["key1"])
        
        # Missing required key
        assert not agent.validate_input({"key2": "value2"}, ["key1"])

class TestPlannerAgent:
    """Test the planner agent."""
    
    @pytest.mark.asyncio
    async def test_planner_execution(self):
        """Test planner agent execution."""
        
        planner = PlannerAgent()
        
        test_text = """
        Abstract: This is a test paper about machine learning.
        
        Introduction: Machine learning is important.
        
        Methodology: We used neural networks.
        
        Results: The results were good.
        
        Conclusion: Machine learning works.
        """
        
        result = await planner.execute({"text": test_text})
        
        assert result.success
        assert "plan" in result.data
        assert "sections" in result.data["plan"]
        assert "strategy" in result.data["plan"]
    
    @pytest.mark.asyncio
    async def test_planner_missing_input(self):
        """Test planner with missing input."""
        
        planner = PlannerAgent()
        
        result = await planner.execute({})
        
        assert not result.success
        assert "Missing required 'text' field" in result.error

class TestExtractorAgent:
    """Test the extractor agent."""
    
    @pytest.mark.asyncio
    async def test_extractor_with_text_content(self):
        """Test extractor with text content."""
        extractor = ExtractorAgent()
        # Use a minimal valid PDF for testing
        def minimal_pdf_bytes():
            return (b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
                    b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n'
                    b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n'
                    b'xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n'
                    b'0000000100 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n147\n%%EOF')
        test_content = minimal_pdf_bytes()
        result = await extractor.execute({"file_content": test_content})
        assert result.success
        assert "text" in result.data
        assert "metadata" in result.data
    
    @pytest.mark.asyncio
    async def test_extractor_missing_input(self):
        """Test extractor with missing input."""
        
        extractor = ExtractorAgent()
        
        result = await extractor.execute({})
        
        assert not result.success
        assert "Missing required" in result.error

class TestSummarizerAgent:
    """Test the summarizer agent."""
    
    @pytest.mark.asyncio
    async def test_summarizer_fallback_mode(self):
        """Test summarizer in fallback mode (no model loaded)."""
        
        summarizer = SummarizerAgent()
        
        test_text = """
        This is a test paper about machine learning. Machine learning is a subset of artificial intelligence.
        It involves training algorithms on data to make predictions. The field has grown rapidly in recent years.
        Applications include image recognition, natural language processing, and recommendation systems.
        """
        
        result = await summarizer.execute({"text": test_text})
        
        assert result.success
        assert "summary" in result.data
        assert "full_text" in result.data["summary"]
    
    @pytest.mark.asyncio
    async def test_summarizer_missing_input(self):
        """Test summarizer with missing input."""
        
        summarizer = SummarizerAgent()
        
        result = await summarizer.execute({})
        
        assert not result.success
        assert "Missing required 'text' field" in result.error

class TestBibliographyAgent:
    """Test the bibliography agent."""
    
    @pytest.mark.asyncio
    async def test_bibliography_extraction(self):
        """Test citation extraction."""
        
        bibliography = BibliographyAgent()
        
        test_text = """
        This paper builds on previous work [1] and recent advances (Smith, 2020).
        The methodology follows Johnson et al. (2021) and uses techniques from [2].
        
        References:
        [1] Brown, A. (2019). Machine Learning Basics.
        [2] Davis, B. (2020). Advanced Algorithms.
        """
        
        citation_plan = {
            "citation_style": "mixed",
            "has_reference_section": True
        }
        
        result = await bibliography.execute({
            "text": test_text,
            "citation_plan": citation_plan
        })
        
        assert result.success
        assert "citations" in result.data
        assert "bibliographies" in result.data
        assert result.data["citation_count"] > 0
    
    @pytest.mark.asyncio
    async def test_bibliography_missing_input(self):
        """Test bibliography with missing input."""
        
        bibliography = BibliographyAgent()
        
        result = await bibliography.execute({})
        
        assert not result.success
        assert "Missing required 'text' field" in result.error

if __name__ == "__main__":
    pytest.main([__file__])
