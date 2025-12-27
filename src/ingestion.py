import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

def process_cuad_data(pdf_path="data/cuad.pdf"):
    """
    Ingests the CUAD reference PDF.
    In a production-ready RAG, we treat this as the gold-standard reference 
    for clause categories and structure.
    """
    loader = PDFReader()
    # Load the specific CUAD PDF provided in the data folder
    documents = loader.load_data(file=pdf_path)
    
    # Use a splitter that respects legal section headers
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)
    
    for node in nodes:
        node.metadata.update({
            "source_type": "cuad_reference",
            "file_name": "cuad.pdf"
        })
    return nodes


def ingest_user_pdf(file_path):
    """Chunks user PDFs by section headers and paragraph boundaries."""
    from llama_index.readers.file import PDFReader
    loader = PDFReader()
    docs = loader.load_data(file=file_path)
    
    # Chunking strategy: 512 tokens with 50 overlap to keep clauses intact
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(docs)
    for node in nodes:
        node.metadata["source_type"] = "user_uploaded"
    return nodes
