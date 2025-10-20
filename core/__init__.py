# ============================================================================
# CORE MODULE - Initialization
# ============================================================================
# Bu dosya 'core' klasörünü bir Python modülü olarak tanımlar.
# İçeriği boş olabilir, sadece varlığı önemlidir.

"""
MentorMate Core Module

Bu modül RAG (Retrieval Augmented Generation) pipeline'ının 
temel bileşenlerini içerir.

Kullanım:
    from core.rag_pipeline import RAGPipeline
    from core.rag_pipeline import validate_answer, preprocess_query
"""

__version__ = "1.0.0"
__author__ = "Onur Tilki"

# Modül içinden otomatik import (opsiyonel)
# Bu satırlar sayesinde şöyle kullanabilirsiniz:
# from core import RAGPipeline
from .rag_pipeline import RAGPipeline, validate_answer, preprocess_query

__all__ = [
    "RAGPipeline",
    "validate_answer", 
    "preprocess_query"
]