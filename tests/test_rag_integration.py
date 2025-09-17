import pytest
from src.rag.vector_store import PaperVectorStore

@pytest.mark.asyncio
async def test_rag_retrieval():
    # Setup: Create a vector store and add a known paper
    vector_store = PaperVectorStore({'persist_directory': 'data/vector_store_test'})
    paper = {
        "id": "test_paper_1",
        "content": "Deep learning for natural language processing.",
        "title": "Test Paper on NLP",
        "authors": "Jane Doe",
        "year": 2023,
        "venue": "TestConf"
    }
    vector_store.add_papers_batch([paper])

    # Query for a related topic
    results = vector_store.search_similar_papers("natural language", n_results=1)
    assert results, "No results retrieved from vector store"
    assert results[0]['id'] == "test_paper_1"
    print("RAG retrieval test passed: ", results[0])
