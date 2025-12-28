

"""
MentorMate Core Module

Bu modül RAG (Retrieval Augmented Generation) pipeline'ının 
temel bileşenlerini içerir.

Kullanım:
    from core.rag_pipeline import RAGPipeline
    from core.rag_pipeline import validate_answer, preprocess_query
"""

__version__ = "1.0.3"
__author__ = "Onur Tilki"

from .rag_pipeline import RAGPipeline, validate_answer, preprocess_query

__all__ = [
    "RAGPipeline",
    "validate_answer", 
    "preprocess_query"
]
