"""
Streamlit web interface for the Research Paper Summarization AI Agent.
"""

import streamlit as st
import asyncio
import json
import tempfile
import os
from typing import Dict, Any
import logging
import numpy as np
from src.agents.orchestrator import PaperSummarizationOrchestrator
from src.evaluation.metrics import SummarizationEvaluator
from src.rag.vector_store import PaperVectorStore
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Research Paper Summarizer AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = PaperSummarizationOrchestrator()
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = SummarizationEvaluator()
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = PaperVectorStore()
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

def main():
    """Main application interface."""
    
    # Title and description
    st.title("🤖 Research Paper Summarizer AI Agent")
    st.markdown("""
    An intelligent multi-agent system that automatically analyzes research papers, generates structured summaries, 
    and manages bibliographies using fine-tuned language models and RAG integration.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📄 Paper Processing", "📊 Evaluation Dashboard", "🔍 Paper Search", "⚙️ System Status"]
    )
    
    if page == "📄 Paper Processing":
        paper_processing_page()
    elif page == "📊 Evaluation Dashboard":
        evaluation_dashboard_page()
    elif page == "🔍 Paper Search":
        paper_search_page()
    elif page == "⚙️ System Status":
        system_status_page()

def paper_processing_page():
    """Main paper processing interface."""
    
    st.header("📄 Paper Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a research paper (PDF or TXT)",
        type=['pdf', 'txt'],
        help="Upload a research paper to generate an automatic summary and bibliography"
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processing Options")
        extraction_method = st.selectbox(
            "PDF Extraction Method",
            ["pdfplumber", "pymupdf", "pypdf2"],
            help="Choose the PDF extraction method"
        )
        
        include_rag = st.checkbox(
            "Enable RAG Context Retrieval",
            value=True,
            help="Use similar papers from the vector database for context"
        )
    
    with col2:
        st.subheader("Output Options")
        summary_format = st.selectbox(
            "Summary Format",
            ["Structured", "Full Text", "Key Points Only"],
            help="Choose the format for the generated summary"
        )
        
        citation_style = st.selectbox(
            "Citation Style",
            ["APA", "MLA", "IEEE", "Chicago"],
            help="Choose the citation format for the bibliography"
        )
    
    # Process button
    if st.button("🚀 Process Paper", type="primary"):
        if uploaded_file is not None:
            process_uploaded_paper(uploaded_file, extraction_method, include_rag, summary_format, citation_style)
        else:
            st.error("Please upload a paper first!")
    
    # Display processing history
    display_processing_history()

def process_uploaded_paper(uploaded_file, extraction_method, include_rag, summary_format, citation_style):
    """Process the uploaded paper."""
    import logging
    with st.spinner("Processing paper... This may take a few minutes."):
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Configure orchestrator
            config = {
                'extractor': {'method': extraction_method},
                'summarizer': {'format': summary_format.lower()},
                'bibliography': {'style': citation_style.lower()}
            }

            rag_context = None
            # Extract text for RAG retrieval if enabled
            if include_rag:
                # Use extractor agent directly to get text
                extractor = st.session_state.orchestrator.extractor
                extraction_result = asyncio.run(extractor.execute({'file_path': tmp_file_path}))
                if extraction_result.success:
                    extracted_text = extraction_result.data['text']
                    # Retrieve similar papers
                    rag_context = st.session_state.vector_store.search_similar_papers(extracted_text, n_results=3)
                    logging.info(f"RAG retrieved {len(rag_context)} contexts: {[c['id'] for c in rag_context]}")
                else:
                    logging.warning("RAG enabled but failed to extract text for retrieval context.")

            # Process paper (pass rag_context if available)
            result = asyncio.run(
                st.session_state.orchestrator.process_paper({
                    'file_path': tmp_file_path,
                    'rag_context': rag_context
                })
            )

            # Clean up temporary file
            os.unlink(tmp_file_path)

            if result['success']:
                display_processing_results(result, uploaded_file.name)
                # Add to history
                st.session_state.processing_history.append({
                    'filename': uploaded_file.name,
                    'timestamp': pd.Timestamp.now(),
                    'success': True,
                    'result': result
                })
                # Add to vector store if RAG is enabled
                if include_rag:
                    add_to_vector_store(result, uploaded_file.name)
            else:
                st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
                st.session_state.processing_history.append({
                    'filename': uploaded_file.name,
                    'timestamp': pd.Timestamp.now(),
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
        except Exception as e:
            st.error(f"An error occurred: {e}")

def display_processing_results(result: Dict[str, Any], filename: str):
    """Display the processing results."""
    
    st.success("✅ Paper processed successfully!")
    
    # Summary section
    st.subheader("📋 Generated Summary")
    
    summary_data = result['summary']['summary']
    
    # Display summary based on type
    if summary_data['type'] == 'structured':
        for section_name, section_content in summary_data['sections'].items():
            with st.expander(f"📖 {section_name.title()}", expanded=True):
                st.write(section_content)
    else:
        st.write(summary_data['full_text'])
    
    # Key points
    if 'key_points' in summary_data:
        st.subheader("🔑 Key Points")
        for i, point in enumerate(summary_data['key_points'], 1):
            st.write(f"{i}. {point}")
    
    # Bibliography section
    st.subheader("📚 Bibliography")
    
    bibliography_data = result['bibliography']
    
    if bibliography_data['citations']:
        # Display citations
        st.write(f"**Found {bibliography_data['citation_count']} citations:**")
        
        citation_df = pd.DataFrame(bibliography_data['citations'])
        st.dataframe(citation_df, use_container_width=True)
        
        # Display formatted bibliography
        for style, formatted_citations in bibliography_data['bibliographies'].items():
            if formatted_citations:
                with st.expander(f"📖 {style.upper()} Format"):
                    for citation in formatted_citations:
                        st.write(f"• {citation}")
    else:
        st.info("No citations found in the paper.")
    
    # Processing metadata
    st.subheader("📊 Processing Metadata")
    
    metadata = result['processing_metadata']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Text Length", f"{metadata.get('text_length', 0):,} chars")
    
    with col2:
        st.metric("Summary Length", f"{metadata.get('summary_length', 0):,} chars")
    
    with col3:
        compression_ratio = metadata.get('compression_ratio', 0)
        st.metric("Compression Ratio", f"{compression_ratio:.1%}")
    
    # Download options
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download summary as text
        summary_text = summary_data['full_text']
        st.download_button(
            "📄 Download Summary",
            summary_text,
            file_name=f"{filename}_summary.txt",
            mime="text/plain"
        )
    
    with col2:
        # Download full results as JSON
        results_json = json.dumps(result, indent=2, default=str)
        st.download_button(
            "📊 Download Full Results",
            results_json,
            file_name=f"{filename}_results.json",
            mime="application/json"
        )

def add_to_vector_store(result: Dict[str, Any], filename: str):
    """Add processed paper to vector store."""
    
    try:
        # Extract content and metadata
        content = result['summary']['full_text']
        metadata = {
            'filename': filename,
            'title': result['paper_metadata'].get('title', filename),
            'author': result['paper_metadata'].get('author', 'Unknown'),
            'processing_date': pd.Timestamp.now().isoformat(),
            'citation_count': result['bibliography']['citation_count']
        }
        
        # Add to vector store
        paper_id = f"{filename}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        success = st.session_state.vector_store.add_paper(paper_id, content, metadata)
        
        if success:
            st.info("📚 Paper added to knowledge base for future RAG retrieval")
        
    except Exception as e:
        st.warning(f"Could not add paper to knowledge base: {str(e)}")

def display_processing_history():
    """Display processing history."""
    
    if st.session_state.processing_history:
        st.subheader("📈 Processing History")
        
        # Create history dataframe
        history_data = []
        for item in st.session_state.processing_history[-10:]:  # Last 10 items
            history_data.append({
                'Filename': item['filename'],
                'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Status': '✅ Success' if item['success'] else '❌ Failed',
                'Summary Length': len(item.get('result', {}).get('summary', {}).get('full_text', '')) if item['success'] else 0
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)

def evaluation_dashboard_page():
    """Evaluation dashboard page."""
    
    st.header("📊 Evaluation Dashboard")
    
    if not st.session_state.processing_history:
        st.info("No papers processed yet. Process some papers to see evaluation metrics.")
        return
    
    # Filter successful processing results
    successful_results = [
        item for item in st.session_state.processing_history 
        if item['success']
    ]
    
    if not successful_results:
        st.info("No successful processing results to evaluate.")
        return
    
    # Overall statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", len(st.session_state.processing_history))
    
    with col2:
        success_rate = len(successful_results) / len(st.session_state.processing_history)
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col3:
        avg_summary_length = np.mean([
            len(item['result']['summary']['summary']['full_text'])
            for item in successful_results
        ])
        st.metric("Avg Summary Length", f"{avg_summary_length:.0f} chars")
    
    with col4:
        avg_citations = np.mean([
            item['result']['bibliography']['citation_count'] 
            for item in successful_results
        ])
        st.metric("Avg Citations", f"{avg_citations:.1f}")
    
    # Processing time analysis
    st.subheader("⏱️ Processing Analysis")
    
    # Create charts
    chart_data = []
    for item in successful_results:
        result = item['result']
        chart_data.append({
            'Filename': item['filename'][:20] + '...' if len(item['filename']) > 20 else item['filename'],
            'Summary Length': len(result['summary']['summary']['full_text']),
            'Citation Count': result['bibliography']['citation_count'],
            'Compression Ratio': result['processing_metadata'].get('compression_ratio', 0)
        })
    
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Summary length distribution
            fig = px.bar(
                chart_df, 
                x='Filename', 
                y='Summary Length',
                title='Summary Length by Paper'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Citation count distribution
            fig = px.bar(
                chart_df, 
                x='Filename', 
                y='Citation Count',
                title='Citations Found by Paper'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def paper_search_page():
    """Paper search and RAG interface."""
    
    st.header("🔍 Paper Search & Knowledge Base")
    
    # Search interface
    search_query = st.text_input(
        "Search for similar papers:",
        placeholder="Enter keywords or research topics..."
    )
    
    if search_query:
        with st.spinner("Searching knowledge base..."):
            try:
                results = st.session_state.vector_store.search_similar_papers(
                    query=search_query,
                    n_results=5
                )
                
                if results:
                    st.subheader(f"📚 Found {len(results)} similar papers:")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Paper {i}: {result['metadata'].get('title', 'Unknown Title')} (Similarity: {result['similarity_score']:.3f})"):
                            st.write("**Content Preview:**")
                            st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
                            
                            st.write("**Metadata:**")
                            metadata_df = pd.DataFrame([result['metadata']])
                            st.dataframe(metadata_df, use_container_width=True)
                else:
                    st.info("No similar papers found in the knowledge base.")
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    
    # Knowledge base statistics
    st.subheader("📊 Knowledge Base Statistics")
    
    try:
        stats = st.session_state.vector_store.get_collection_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Papers", stats.get('total_papers', 0))
        
        with col2:
            st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
            
    except Exception as e:
        st.error(f"Could not retrieve knowledge base statistics: {str(e)}")

def system_status_page():
    """System status and configuration page."""
    
    st.header("⚙️ System Status")
    
    # Agent status
    st.subheader("🤖 Agent Status")
    
    try:
        agent_status = st.session_state.orchestrator.get_agent_status()
        
        for agent_name, status in agent_status.items():
            with st.expander(f"🔧 {agent_name.title()} Agent"):
                status_df = pd.DataFrame([status])
                st.dataframe(status_df, use_container_width=True)
                
    except Exception as e:
        st.error(f"Could not retrieve agent status: {str(e)}")
    
    # System configuration
    st.subheader("⚙️ Configuration")
    
    with st.expander("📝 Current Configuration"):
        config = {
            "Extraction Methods": ["pdfplumber", "pymupdf", "pypdf2"],
            "Citation Styles": ["APA", "MLA", "IEEE", "Chicago"],
            "Summary Formats": ["Structured", "Full Text", "Key Points Only"],
            "RAG Enabled": True,
            "Vector Store": "ChromaDB",
            "Embedding Model": "all-MiniLM-L6-v2"
        }
        
        config_df = pd.DataFrame(list(config.items()), columns=['Setting', 'Value'])
        
        # Ensure all values in the Value column are strings if they are lists
        if "Value" in config_df.columns:
            config_df["Value"] = config_df["Value"].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        st.dataframe(config_df, use_container_width=True)
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    
    if st.session_state.processing_history:
        successful_count = sum(1 for item in st.session_state.processing_history if item['success'])
        total_count = len(st.session_state.processing_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", f"{successful_count/total_count:.1%}")
        
        with col2:
            st.metric("Total Processed", total_count)
        
        with col3:
            st.metric("Knowledge Base Size", st.session_state.vector_store.get_collection_stats().get('total_papers', 0))

if __name__ == "__main__":
    main()
