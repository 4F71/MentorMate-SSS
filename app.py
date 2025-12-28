import streamlit as st
import os
import asyncio
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from core.rag_pipeline import RAGPipeline, validate_answer, preprocess_query



load_dotenv()
asyncio.set_event_loop(asyncio.new_event_loop())

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "mentormate_faq"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_FILES = [
    os.path.join(PROJECT_ROOT, "data", "enriched_dataset.jsonl"),
    os.path.join(PROJECT_ROOT, "data", "generated_data_google.jsonl")
]

EXPERT_MODE = {
    "name": "Hibrit Mod",  
    "icon": "🎯",
    "color":"#1f77b4",
    "desc": "Veritabanı + Genel sorular için güvenli LLM desteği."  
}


def load_jsonl_data(file_path: str) -> list:
    documents = []
    
    if not os.path.exists(file_path):
        return documents
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
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
            except:
                continue
    
    return documents


def create_database_runtime():
    
    all_documents = []
    for file_path in DATA_FILES:
        docs = load_jsonl_data(file_path)
        all_documents.extend(docs)
    
    if not all_documents:
        raise Exception("Veri dosyaları yüklenemedi!")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 32}
    )
    
    vectordb = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    return vectordb


def check_and_setup_database():
    if not os.path.exists(DB_PATH):
        with st.spinner(" İlk kurulum yapılıyor... (~2-3 dakika)"):
            try:
                st.info(" Veritabanı oluşturuluyor...")
                create_database_runtime()
                st.success(" Veritabanı başarıyla oluşturuldu!")
                st.balloons()
            except Exception as e:
                st.error(f" Veritabanı oluşturulamadı: {str(e)}")
                st.stop()


@st.cache_resource(show_spinner=False, hash_funcs={type: id})  
def load_rag_pipeline(_force_reload=False): 
    """RAG Pipeline'ı yükler - Cache fix ile"""
    if not GOOGLE_API_KEY:
        st.error(" Google API anahtarı bulunamadı! Lütfen Secrets'a ekleyin.")
        st.stop()
    
    check_and_setup_database()
    
    try:
        pipeline = RAGPipeline(
            google_api_key=GOOGLE_API_KEY,
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL,
            llm_model="gemini-2.0-flash",
            temperature=0.01
        )
        return pipeline
    except Exception as e:
        st.error(f" RAG Pipeline yüklenemedi: {str(e)}")
        st.stop()



def main():
    st.set_page_config(
        page_title="MentorMate Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""<style>.stButton > button {width: 100%;}</style>""", unsafe_allow_html=True)
    
    if "force_reload" not in st.session_state:
        st.session_state.force_reload = False
    
    pipeline = load_rag_pipeline(_force_reload=st.session_state.force_reload)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Merhaba! Ben MentorMate. Bootcamp hakkında sorularınızı yanıtlamak için buradayım."
        }]
    
    with st.sidebar:
        st.title(" MentorMate")
        
        st.markdown(f"""
        <div style='padding: 15px; background-color: {EXPERT_MODE['color']}22; 
        border-radius: 8px; border-left: 4px solid {EXPERT_MODE['color']}'>
        <h3 style='margin:0; color: {EXPERT_MODE['color']}'>{EXPERT_MODE['icon']} {EXPERT_MODE['name']}</h3>
        <p style='margin:5px 0 0 0; font-size:0.9em'>{EXPERT_MODE['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Sohbet İstatistikleri")
        user_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_count = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sorular", user_count)
        with col2:
            st.metric("Cevaplar", bot_count)
        
        st.markdown("---")
        
        if st.button(" Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Merhaba! Ben MentorMate. Bootcamp hakkında sorularınızı yanıtlamak için buradayım. 🚀"
            }]
            pipeline.clear_memory()
            st.rerun()
        
        if st.button(" Sistemi Yenile", use_container_width=True, help="Kod güncellemelerini yükler"):
            st.session_state.force_reload = True
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("###  Veritabanı")
        stats = pipeline.get_stats()
        st.caption(f" `{os.path.basename(stats['db_path'])}/`")
        st.caption(f"`{stats['embedding_model'].split('/')[-1][:35]}`")
        st.caption(f" Mod: {stats.get('mode', 'Uzman')}")  
        
        st.markdown("---")
        st.markdown("[ GitHub Repo](https://github.com/4F71/MentorMate-SSS)")
    
    st.title(f"{EXPERT_MODE['icon']} MentorMate Chatbot")
    st.caption("Bootcamp hakkında **güvenilir bilgiler** ve **genel sorulara** yanıt veriyorum.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if user_input := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner(f"{EXPERT_MODE['icon']} Düşünüyorum..."):
                try:
                    enriched_query = preprocess_query(user_input)
                    result = pipeline.query(enriched_query)
                    final_answer = result.get("answer", "Bir hata oluştu.").strip()
                    
                    st.markdown(final_answer)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_answer
                    })
                except Exception as e:
                    error_msg = f" Bir hata oluştu: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()