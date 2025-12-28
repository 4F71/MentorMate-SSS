#!/usr/bin/env python3

"""
Kullanım:
    python setup_database.py
"""

import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "mentormate_faq"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATA_FILES = [
    os.path.join(PROJECT_ROOT, "data", "enriched_dataset.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "generated_data_google.jsonl")
]


def load_data_from_jsonl(file_path: str) -> list:
    documents = []
    
    if not os.path.exists(file_path):
        print(f" Dosya bulunamadı: {file_path}")
        return documents
    
    print(f"Yükleniyor: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                question = data.get("question", "")
                answer = data.get("answer", "")
                
                if question and answer:
                    page_content = f"Soru: {question}\nCevap: {answer}"
                    doc = Document(
                        page_content=page_content,
                        metadata={"source": os.path.basename(file_path)}
                    )
                    documents.append(doc)
            except json.JSONDecodeError:
                print(f"  Satır {i} atlandı (JSON hatası)")
                continue

    print(f"  {len(documents)} doküman yüklendi\n")
    return documents


def create_database():

    print("="*70)
    print(" MentorMate - ChromaDB Kurulum Scripti")
    print("="*70)
    print()
    
    if os.path.exists(DB_PATH):
        response = input(f"  '{DB_PATH}' zaten mevcut. Yeniden oluşturulsun mu? (y/n): ")
        if response.lower() != 'y':
            print(" İşlem iptal edildi.")
            return
        
        print("  Mevcut veritabanı siliniyor...")
        import shutil
        shutil.rmtree(DB_PATH)
        print("  Silindi\n")
    
    print(" Veri dosyaları yükleniyor...")
    all_documents = []
    
    for file_path in DATA_FILES:
        docs = load_data_from_jsonl(file_path)
        all_documents.extend(docs)
    
    if not all_documents:
        print(" HATA: Hiçbir veri yüklenemedi!")
        print("   Lütfen data/ klasöründe şu dosyaların olduğundan emin olun:")
        for f in DATA_FILES:
            print(f"   - {os.path.basename(f)}")
        return
    
    print(f" Toplam {len(all_documents)} doküman yüklendi")
    print()
    
    print(" Embedding modeli yükleniyor...")
    print(f"   Model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 32}
    )
    print("   Model yüklendi\n")
    
    print("  ChromaDB veritabanı oluşturuluyor...")
    print("   (Bu işlem birkaç dakika sürebilir...)")
    
    try:
        vectordb = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME
        )
        
        print("  Veritabanı oluşturuldu\n")
        

        print(" Veritabanı doğrulanıyor...")
        collection_count = vectordb._collection.count()
        print(f"  {collection_count} doküman veritabanında")
        
        print("\n Test sorgusu yapılıyor...")
        results = vectordb.similarity_search("bootcamp sertifika", k=1)
        if results:
            print("   Test başarılı!")
            print(f"  İlk sonuç: {results[0].page_content[:100]}...")
        
    except Exception as e:
        print(f" HATA: {e}")
        return
    
    print()
    print("="*70)
    print(" KURULUM TAMAMLANDI!")
    print("="*70)
    print()
    print("Şimdi uygulamayı çalıştırabilirsiniz:")
    print("  streamlit run app.py")
    print()



if __name__ == "__main__":
    try:
        create_database()
    except KeyboardInterrupt:
        print("\n\n İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n Beklenmeyen hata: {e}")