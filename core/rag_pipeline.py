import os
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain



EXPERT_PROMPT_TEMPLATE = """Sen MentorMate adlı bootcamp uzman asistanısın. SADECE verilen dokümanları kullanarak cevap veriyorsun.

KRİTİK KURALLAR:
1. Cevabını SADECE aşağıdaki DOKÜMANLAR'dan al
2. Dokümanlarda cevap YOKSA: "Bu konuda veri setimde bilgi bulunmuyor."
3. Kendi bilgini ASLA KULLANMA
4. Kısa, net, profesyonel ve DOĞAL bir dille cevap ver
5. ASLA kaynak, dosya adı veya meta bilgi ekleme
6. AYNI SORUYA HER ZAMAN AYNI CEVABI VER (tutarlılık önemli)
7. Cevabını tek seferde ver, tekrar etme

DOKÜMANLAR:
{context}

SORU: {question}

CEVAP:"""

# YENİ: Genel sorular için LLM promptu
GENERAL_LLM_PROMPT = """Sen MentorMate adlı yardımcı bir asistansın. Bu soru bootcamp veritabanında yok ama genel bir soru.

KRİTİK KURALLAR:
1. Sadece GENEL BİLGİ gerektiren sorulara cevap ver
2. Bootcamp-spesifik bilgi ASLA uydurma (tarih, süre, kurallar vb.)
3. Kısa, doğal ve yardımcı ol
4. Emin değilsen "Bu konuda emin değilim" de

SORU: {question}

CEVAP:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Sohbet geçmişi ve yeni soruyu kullanarak, ANAHTAR KELİMELERİ İÇEREN tek başına anlaşılır bir arama sorgusu oluştur.

ÖNEMLİ KURALLAR:
1. BÜYÜK/küçük harf farkı gözetme - "Sertifika" = "sertifika" = "SERTİFİKA"
2. Soru işareti olsun/olmasın aynı anlama gelen soruları birleştir
3. Anahtar kelimeleri MUTLAKA koru (örn: "katılım oranı", "sertifika", "bootcamp süresi")
4. Eş anlamlı kelimeleri ekle (örn: "iştirak" = "katılım", "web semineri" = "canlı yayın")
5. Tüm kelimeler küçük harfle yazılmalı

SOHBET GEÇMİŞİ:
{chat_history}

YENİ SORU:
{question}

ANAHTAR KELİME ZENGİN SORGU (küçük harfle):""")



def categorize_question(question: str) -> str:
    """
    Soruyu kategorize eder ve güvenli LLM kullanımına karar verir
    
    Returns:
        "bootcamp_specific": Bootcamp hakkında - HALÜSİNASYON RİSKİ!
        "general_safe": Genel bilgi - LLM kullanılabilir
        "greeting": Selamlama - Direkt cevap
    """
    q_lower = question.lower()
    
    greetings = ["merhaba", "selam", "hey", "hi", "günaydın", "iyi günler"]
    if any(g in q_lower for g in greetings):
        return "greeting"
    
    bootcamp_keywords = [
        "bootcamp", "sertifika", "katılım", "mentor", "proje", "grup",
        "canlı yayın", "webinar", "akbank", "eğitim süresi", "tarih",
        "toplantı", "zulip", "github repo", "teslim", "döküman"
    ]
    if any(kw in q_lower for kw in bootcamp_keywords):
        return "bootcamp_specific"
    
    general_safe_patterns = [
        "nedir", "ne demek", "nasıl", "kimdir", "matematik", "hesapla",
        "mentormate nedir", "sen kimsin", "ne yaparsın", "+", "-", "*", "/"
    ]
    if any(pattern in q_lower for pattern in general_safe_patterns):
        return "general_safe"
    
    return "bootcamp_specific"



class RAGPipeline:
    """RAG sistemi için merkezi yönetim sınıfı - Hibrit Mod"""
    
    def __init__(
        self,
        google_api_key: str,
        db_path: str,
        collection_name: str = "mentormate_faq",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gemini-2.0-flash",
        temperature: float = 0.01
    ):
        self.google_api_key = google_api_key
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.temperature = temperature
        
        self.llm = None
        self.llm_general = None  
        self.embeddings = None
        self.vectordb = None
        self.retriever = None
        self.memory = None
        self.chain = None
        
        self._initialize()
    
    def _initialize(self):
        self._setup_llm()
        self._setup_embeddings()
        self._setup_vectordb()
        self._setup_retriever()
        self._setup_memory()
        self._setup_chain()
    
    def _setup_llm(self):
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model_name,
            google_api_key=self.google_api_key,
            temperature=self.temperature
        )
        
        self.llm_general = ChatGoogleGenerativeAI(
            model=self.llm_model_name,
            google_api_key=self.google_api_key,
            temperature=0.3  
        )
    
    def _setup_embeddings(self):
        """Embedding modelini yükler"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 32}
        )
    
    def _setup_vectordb(self):
        """Vector database'i yükler"""
        self.vectordb = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
    
    def _setup_retriever(self):
        """MultiQuery Retriever'ı kurar"""
        base_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                'k': 5,
                'fetch_k': 25,
                'lambda_mult': 0.6
            }
        )
        
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            include_original=True
        )
    
    def _setup_memory(self):
        """Conversation memory'yi başlatır"""
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True,
            output_key='answer'
        )
    
    def _setup_chain(self):
        """Konuşma zincirini oluşturur"""
        prompt = PromptTemplate(
            template=EXPERT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )
    
    def query(self, question: str) -> Dict:
        """
        YENİ: Hibrit sorgu işleme
        1. Önce RAG'e sor
        2. Cevap güvensizse ve soru güvenli kategorideyse → LLM'e sor
        3. Bootcamp-spesifik sorularda → "Bilgi yok" de
        """
        try:
            category = categorize_question(question)
            
            if category == "greeting":
                return {
                    "answer": "Merhaba! Ben MentorMate. Size nasıl yardımcı olabilirim?",
                    "source_documents": []
                }
            
            result = self.chain.invoke({"question": question})
            answer = result.get("answer", "").strip()
            source_docs = result.get("source_documents", [])
            
            is_confident = self._check_confidence(answer, source_docs)
            
            if not is_confident and category == "general_safe":
                return self._general_llm_fallback(question)
            
            if not is_confident and category == "bootcamp_specific":
                return {
                    "answer": " Bu konuda veri setimde güvenilir bilgi bulunmuyor.",
                    "source_documents": source_docs
                }
            
            # Normal RAG cevabı
            return result
            
        except Exception as e:
            raise Exception(f"Query işleme hatası: {str(e)}")
    
    def _check_confidence(self, answer: str, source_docs: List) -> bool:
        """
        Cevabın güvenilir olup olmadığını kontrol eder
        """
        answer_lower = answer.lower()
        
        no_info_keywords = ["veri setimde", "bilgi bulunmuyor", "bilgim yok"]
        if any(kw in answer_lower for kw in no_info_keywords):
            return False
        
        if not source_docs:
            return False
        
        answer_words = set([w for w in answer_lower.split() if len(w) > 3])
        if not answer_words:
            return True
        
        source_text = " ".join([doc.page_content.lower() for doc in source_docs])
        matched_words = [w for w in answer_words if w in source_text]
        overlap_ratio = len(matched_words) / len(answer_words)
        
        return overlap_ratio >= 0.20
    
    def _general_llm_fallback(self, question: str) -> Dict:
        """
        YENİ: Genel sorular için güvenli LLM fallback
        """
        prompt = PromptTemplate(
            template=GENERAL_LLM_PROMPT,
            input_variables=["question"]
        )
        
        try:
            formatted_prompt = prompt.format(question=question)
            response = self.llm_general.invoke(formatted_prompt)
            answer = response.content.strip()
            
            return {
                "answer": answer,
                "source_documents": []
            }
        except Exception as e:
            return {
                "answer": "⚠️ Bu konuda size yardımcı olamıyorum.",
                "source_documents": []
            }
    
    def clear_memory(self):
        """Sohbet geçmişini temizler"""
        self.memory.clear()
    
    def get_stats(self) -> Dict:
        """Pipeline istatistiklerini döner"""
        return {
            "llm_model": self.llm_model_name,
            "embedding_model": self.embedding_model_name,
            "temperature": self.temperature,
            "collection_name": self.collection_name,
            "db_path": self.db_path,
            "mode": "Hibrit (RAG + Güvenli LLM Fallback)"
        }





def validate_answer(answer: str, source_docs: List) -> str:

    if not source_docs:
        return " Bu konuda veri setimde güvenilir bilgi bulunmuyor."
    
    answer_lower = answer.lower()
    
    no_info_keywords = ["veri setimde", "bilgi bulunmuyor", "bilgim yok"]
    if any(keyword in answer_lower for keyword in no_info_keywords):
        return answer
    
    answer_words = set([w for w in answer_lower.split() if len(w) > 3])
    source_text = " ".join([doc.page_content.lower() for doc in source_docs])
    
    if not answer_words:
        return answer
    
    matched_words = [w for w in answer_words if w in source_text]
    overlap_ratio = len(matched_words) / len(answer_words)
    
    if overlap_ratio < 0.20:
        return " Bu konuda veri setimde güvenilir bilgi bulunmuyor."
    
    return answer


def preprocess_query(query: str) -> str:
    """
    Sorguya anahtar kelime zenginleştirmesi ve normalizasyon yapar
    """
    query_normalized = query.lower()
    
    turkish_chars = {
        'İ': 'i', 'I': 'ı', 'Ğ': 'ğ', 'Ü': 'ü',
        'Ş': 'ş', 'Ö': 'ö', 'Ç': 'ç'
    }
    for upper, lower in turkish_chars.items():
        query_normalized = query_normalized.replace(upper, lower)
    
    keyword_map = {
        "katılım": ["iştirak", "katılım oranı", "yoklama", "attendance", "devam"],
        "canlı yayın": ["webinar", "web semineri", "youtube", "yayın", "live", "stream"],
        "sertifika": ["certificate", "belge", "sertifikadaki", "diploma", "sertifikası"],
        "bootcamp": ["eğitim", "kurs", "program", "kampı", "camp", "training"],
        "süre": ["zaman", "gün", "hafta", "ne kadar", "kaç", "duration"],
        "mentor": ["danışman", "eğitmen", "mentör", "öğretmen"],
        "proje": ["ödev", "task", "görev", "assignment", "project", "tamamlama"],
        "github": ["git", "repo", "repository", "kod yükleme", "arayüz"],
        "grup": ["ekip", "takım", "team", "bireysel", "tek kişi", "iki kişi"],
        "iş": ["staj", "kariyer", "fırsat", "employment", "job"],
        "arşiv": ["kayıt", "video", "recording", "kaydediliyor"],
        "duyuru": ["announcement", "bildirim", "haber", "kanal", "zulip"],
        "takvim": ["tarih", "gün", "program", "schedule", "zamanlama"],
        "toplantı": ["meeting", "buluşma", "görüşme", "saat", "zaman"]
    }
    
    words = query_normalized.split()
    if len(words) <= 2:
        for word in words:
            for keyword, synonyms in keyword_map.items():
                if keyword in word or word in keyword:
                    query_normalized += " " + keyword + " " + " ".join(synonyms)
                    break
    
    enriched_query = query_normalized
    for keyword, synonyms in keyword_map.items():
        if keyword in query_normalized:
            enriched_query += " " + " ".join(synonyms)
    
    return enriched_query