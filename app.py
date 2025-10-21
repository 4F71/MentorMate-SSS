# ============================================================================
# APP.PY: MentorMate Chatbot - Streamlit Web Arayüzü (Otomatik Setup)
# ============================================================================

import streamlit as st
import os
import asyncio
import subprocess
from dotenv import load_dotenv

# Core modüllerden import
from core.rag_pipeline import RAGPipeline, validate_answer, preprocess_query

# ============================================================================
# 1. PROJE YAPILANDIRMASI
# ============================================================================

load_dotenv()
asyncio.set_event_loop(asyncio.new_event_loop())

# Proje yolları
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "mentormate_faq"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Uzman mod ayarları
EXPERT_MODE = {
    "name": "Uzman Mod",
    "icon": "🎯",
    "color": "#1f77b4",
    "desc": "Sadece veritabanındaki güvenilir bilgileri verir. Bilmediği sorularda açıkça belirtir."
}

# ============================================================================
# 2. OTOMATİK VERİTABANI KURULUMU
# ============================================================================

def check_and_setup_database():
    """Veritabanı yoksa otomatik olarak oluşturur"""
    if not os.path.exists(DB_PATH):
        with st.spinner("🔧 İlk kurulum yapılıyor... (Bu işlem ~2-3 dakika sürebilir)"):
            try:
                st.info("📦 Veritabanı bulunamadı, oluşturuluyor...")
                
                # setup_database.py'yi çalıştır
                result = subprocess.run(
                    ["python", "setup_database.py"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 dakika timeout
                )
                
                if result.returncode == 0:
                    st.success("✅ Veritabanı başarıyla oluşturuldu!")
                    st.balloons()
                else:
                    st.error(f"❌ Veritabanı oluşturulamadı: {result.stderr}")
                    st.stop()
                    
            except subprocess.TimeoutExpired:
                st.error("⏰ Veritabanı oluşturma süresi doldu. Lütfen tekrar deneyin.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Beklenmeyen hata: {str(e)}")
                st.stop()

# ============================================================================
# 3. BİLEŞEN YÜKLEME (Cached)
# ============================================================================

@st.cache_resource
def load_rag_pipeline():
    """RAG Pipeline'ı yükler ve cache'ler"""
    if not GOOGLE_API_KEY:
        st.error("⚠️ Google API anahtarı bulunamadı! Lütfen Streamlit Cloud'da 'Secrets' kısmına ekleyin.")
        st.code("""
# Streamlit Cloud Secrets formatı (.streamlit/secrets.toml):
GOOGLE_API_KEY = "your_api_key_here"
        """)
        st.stop()
    
    # Veritabanı kontrolü ve kurulum
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
        st.error(f"❌ RAG Pipeline yüklenemedi: {str(e)}")
        st.stop()

# ============================================================================
# 4. STREAMLIT ARAYÜZÜ
# ============================================================================

def main():
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="MentorMate Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS stilleri
    st.markdown("""
    <style>
    .stButton > button {width: 100%;}
    </style>
    """, unsafe_allow_html=True)
    
    # RAG Pipeline'ı yükle
    pipeline = load_rag_pipeline()
    
    # Session state başlatma
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Merhaba! Ben MentorMate. Bootcamp ile ilgili sorularınızı yanıtlamak için buradayım. 🚀"
        }]
    
    # ============================================================================
    # SIDEBAR
    # ============================================================================
    
    with st.sidebar:
        st.title("⚙️ MentorMate")
        
        # Mod bilgisi
        st.markdown(f"""
        <div style='padding: 15px; background-color: {EXPERT_MODE['color']}22; 
        border-radius: 8px; border-left: 4px solid {EXPERT_MODE['color']}'>
        <h3 style='margin:0; color: {EXPERT_MODE['color']}'>{EXPERT_MODE['icon']} {EXPERT_MODE['name']}</h3>
        <p style='margin:5px 0 0 0; font-size:0.9em'>{EXPERT_MODE['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sohbet istatistikleri
        st.markdown("### 📊 Sohbet İstatistikleri")
        user_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_count = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sorular", user_count)
        with col2:
            st.metric("Cevaplar", bot_count)
        
        st.markdown("---")
        
        # Sohbeti temizle butonu
        if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Merhaba! Ben MentorMate. Bootcamp ile ilgili sorularınızı yanıtlamak için buradayım. 🚀"
            }]
            pipeline.clear_memory()
            st.rerun()
        
        st.markdown("---")
        
        # Veritabanı bilgisi
        st.markdown("### 🗄️ Veritabanı")
        stats = pipeline.get_stats()
        st.caption(f"📁 `{os.path.basename(stats['db_path'])}/`")
        st.caption(f"🤖 `{stats['embedding_model'].split('/')[-1][:35]}`")
        st.caption(f"🔥 Model: `{stats['llm_model']}`")
        st.caption(f"🌡️ Temperature: `{stats['temperature']}`")
        
        # GitHub linki
        st.markdown("---")
        st.markdown("### 🔗 Proje")
        st.markdown("[📦 GitHub Repo](https://github.com/4F71/MentorMate-SSS)")
    
    # ============================================================================
    # ANA İÇERİK
    # ============================================================================
    
    st.title(f"{EXPERT_MODE['icon']} MentorMate Chatbot")
    st.caption("Bootcamp hakkında **sadece güvenilir bilgiler** veriyorum.")
    
    # Mesaj geçmişini göster
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Kullanıcı girişi
    if user_input := st.chat_input("Sorunuzu buraya yazın..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Selamlaşma kontrolü
        greetings = ["merhaba", "selam", "hey", "hi", "günaydın", "iyi günler"]
        if user_input.lower().strip() in greetings:
            response = f"Merhaba! Ben MentorMate, **{EXPERT_MODE['name']}** ile çalışıyorum. Size nasıl yardımcı olabilirim? 😊"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        else:
            # RAG ile soru cevaplama
            with st.chat_message("assistant"):
                with st.spinner(f"{EXPERT_MODE['icon']} Düşünüyorum..."):
                    try:
                        # Sorguyu zenginleştir
                        enriched_query = preprocess_query(user_input)
                        
                        # RAG Pipeline ile cevap al
                        result = pipeline.query(enriched_query)
                        
                        # Cevabı valide et
                        raw_answer = result.get("answer", "Bir hata oluştu.").strip()
                        source_docs = result.get("source_documents", [])
                        final_answer = validate_answer(raw_answer, source_docs)
                        
                        # Cevabı göster
                        st.markdown(final_answer)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_answer
                        })
                    
                    except Exception as e:
                        error_msg = f"❌ Bir hata oluştu: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })

# ============================================================================
# ÇALIŞTIRMA
# ============================================================================

if __name__ == "__main__":
    main()