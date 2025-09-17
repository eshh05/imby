"""
Quick test to verify the system components work.
"""

import asyncio
import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Minimal valid PDF bytes: a one-page blank PDF
def minimal_pdf_bytes():
    return (b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n147\n%%EOF')

@pytest.mark.asyncio
async def test_basic_functionality():
    """Test basic functionality without heavy model loading."""
    
    print("🧪 Quick System Test")
    print("=" * 30)
    
    try:
        # Test 1: Import all modules
        print("1. Testing imports...")
        from src.agents.base_agent import BaseAgent, AgentResult
        from src.agents.planner_agent import PlannerAgent
        from src.agents.extractor_agent import ExtractorAgent
        from src.agents.summarizer_agent import SummarizerAgent
        from src.agents.bibliography_agent import BibliographyAgent
        from src.agents.orchestrator import PaperSummarizationOrchestrator
        print("   ✅ All imports successful")
        
        # Test 2: Test planner agent
        print("2. Testing planner agent...")
        planner = PlannerAgent()
        test_text = """
        Abstract: This is a test paper.
        Introduction: This paper discusses testing.
        Methodology: We used testing methods.
        Results: Tests passed.
        Conclusion: Testing works.
        """
        
        result = await planner.execute({"text": test_text})
        if result.success:
            print("   ✅ Planner agent working")
        else:
            print(f"   ❌ Planner failed: {result.error}")
            return False
        
        # Test 3: Test extractor agent
        print("3. Testing extractor agent...")
        extractor = ExtractorAgent()
        file_content = minimal_pdf_bytes()
        assert file_content.startswith(b'%PDF')
        
        result = await extractor.execute({"file_content": file_content})
        if result.success:
            print("   ✅ Extractor agent working")
        else:
            print(f"   ❌ Extractor failed: {result.error}")
            return False
        
        # Test 4: Test summarizer agent (fallback mode)
        print("4. Testing summarizer agent...")
        summarizer = SummarizerAgent()
        
        result = await summarizer.execute({"text": test_text})
        if result.success:
            print("   ✅ Summarizer agent working (fallback mode)")
        else:
            print(f"   ❌ Summarizer failed: {result.error}")
            return False
        
        # Test 5: Test bibliography agent
        print("5. Testing bibliography agent...")
        bibliography = BibliographyAgent()
        test_text_with_citations = """
        This paper references previous work [1] and recent studies (Smith, 2020).
        References:
        [1] Brown, A. (2019). Test Paper.
        """
        
        result = await bibliography.execute({
            "text": test_text_with_citations,
            "citation_plan": {"citation_style": "mixed"}
        })
        if result.success:
            print("   ✅ Bibliography agent working")
        else:
            print(f"   ❌ Bibliography failed: {result.error}")
            return False
        
        print("\n🎉 All basic tests passed!")
        print("\nSystem is ready for use. Try:")
        print("- streamlit run app.py (for web interface)")
        print("- python main.py process your_paper.pdf (for CLI)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)
