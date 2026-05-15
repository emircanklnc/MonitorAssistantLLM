"""
Monitör RAG Chatbot - Streamlit Arayüzü
Ollama (Qwen 2.5) + ChromaDB + bge-m3 Embedding
"""

import streamlit as st
import time
from rag_system import (
    check_ollama_connection,
    check_model_available,
    get_system_status,
    build_vector_store,
    vector_store_exists,
    generate_response_stream,
    get_collection,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

# ─── Sayfa Konfigürasyonu ────────────────────────────────────────────────────
st.set_page_config(
    page_title="🖥️ Monitör Asistanı",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Ana tema */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Başlık stili */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        font-family: 'Segoe UI', sans-serif;
    }

    .sub-title {
        color: #8892b0;
        text-align: center;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Sidebar stili */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* Durum kartları */
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }

    .status-item {
        display: flex;
        align-items: center;
        padding: 6px 0;
        font-size: 0.9rem;
    }

    .status-ok { color: #00d4aa; }
    .status-err { color: #ff6b6b; }

    /* Chat mesajları */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        margin: 8px 0;
        backdrop-filter: blur(5px);
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }

    /* Butonlar */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }

    /* İstatistik kartları */
    .stat-container {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }

    .stat-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        flex: 1;
        text-align: center;
    }

    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        color: #8892b0;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Hoşgeldin kartı */
    .welcome-card {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 40px auto;
        max-width: 700px;
    }

    .welcome-card h3 {
        color: #ccd6f6;
        margin-bottom: 12px;
    }

    .welcome-card p {
        color: #8892b0;
        line-height: 1.6;
    }

    .example-queries {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: -20px;
        margin-bottom: 20px;
    }

    /* Öneri butonları özel stili */
    div[data-testid="column"] .stButton > button {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 20px !important;
        padding: 6px 12px !important;
        color: #a8b2d1 !important;
        font-size: 0.8rem !important;
        font-weight: 400 !important;
        height: auto !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="column"] .stButton > button:hover {
        background: rgba(102, 126, 234, 0.25) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
        color: #ccd6f6 !important;
        transform: translateY(-1px) !important;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 20px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 3px;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Başlat ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db_ready" not in st.session_state:
    st.session_state.vector_db_ready = vector_store_exists()
if "indexing" not in st.session_state:
    st.session_state.indexing = False
if "suggestion_clicked" not in st.session_state:
    st.session_state.suggestion_clicked = None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖥️ Monitör Asistanı")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Sistem Durumu
    st.markdown("### ⚙️ Sistem Durumu")
    status = get_system_status()

    col1, col2 = st.columns(2)
    with col1:
        if status["ollama_connected"]:
            st.success("Ollama ✓", icon="🟢")
        else:
            st.error("Ollama ✗", icon="🔴")

    with col2:
        if st.session_state.vector_db_ready:
            st.success("Vektör DB ✓", icon="🟢")
        else:
            st.warning("Vektör DB ✗", icon="🟡")

    # Model Durumu
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🤖 Modeller")

    if status["llm_model"]:
        st.markdown(f"✅ **LLM**: `{LLM_MODEL}`")
    else:
        st.markdown(f"❌ **LLM**: `{LLM_MODEL}` bulunamadı")

    if status["embedding_model"]:
        st.markdown(f"✅ **Embedding**: `{EMBEDDING_MODEL}`")
    else:
        st.markdown(f"❌ **Embedding**: `{EMBEDDING_MODEL}` bulunamadı")

    # Veritabanı İstatistikleri
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Veritabanı")

    collection = get_collection()
    if collection:
        doc_count = collection.count()
        st.markdown(f"📄 **Döküman Sayısı**: `{doc_count:,}`")
    else:
        st.markdown("📄 **Döküman Sayısı**: `0`")

    # İndeksleme Butonu
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.button("🔄 Vektör Veritabanını Oluştur / Yenile", use_container_width=True, disabled=st.session_state.indexing):
        if not status["ollama_connected"]:
            st.error("❌ Ollama bağlantısı kurulamadı! Önce Ollama'yı başlatın.")
        elif not status["embedding_model"]:
            st.error(f"❌ `{EMBEDDING_MODEL}` modeli bulunamadı! `ollama pull {EMBEDDING_MODEL}` komutunu çalıştırın.")
        else:
            st.session_state.indexing = True
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.markdown(f"⏳ {message}")

            try:
                total = build_vector_store(progress_callback=update_progress)
                st.session_state.vector_db_ready = True
                st.session_state.indexing = False
                progress_bar.progress(1.0)
                status_text.markdown(f"✅ **{total:,}** monitör başarıyla indekslendi!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.session_state.indexing = False
                st.error(f"❌ Hata: {str(e)}")

    # Sohbet Temizleme
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Bilgi
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #6272a4; font-size: 0.75rem; text-align: center; margin-top: 20px;">
        <p>RAG Chatbot v1.0</p>
        <p>Qwen 2.5 + bge-m3 + ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Ana İçerik ──────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🖥️ Monitör Asistanı</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">RAG Tabanlı Akıllı Monitör Danışmanı • Qwen 2.5 + ChromaDB</p>', unsafe_allow_html=True)

# Hazır değilse uyarı göster
if not status["ollama_connected"]:
    st.error("""
    ⚠️ **Ollama bağlantısı kurulamadı!**

    Ollama'nın çalıştığından emin olun:
    1. Ollama'yı başlatın
    2. Terminal'de `ollama serve` komutunu çalıştırın
    """)
    st.stop()

if not status["llm_model"]:
    st.warning(f"""
    ⚠️ **`{LLM_MODEL}` modeli bulunamadı!**

    Terminalde şu komutu çalıştırın:
    ```
    ollama pull {LLM_MODEL}
    ```
    """)

if not status["embedding_model"]:
    st.warning(f"""
    ⚠️ **`{EMBEDDING_MODEL}` modeli bulunamadı!**

    Terminalde şu komutu çalıştırın:
    ```
    ollama pull {EMBEDDING_MODEL}
    ```
    """)

if not st.session_state.vector_db_ready:
    st.info("ℹ️ Vektör veritabanı henüz oluşturulmamış. Soldaki menüden **'Vektör Veritabanını Oluştur'** butonuna tıklayın.")

# ─── Sohbet Geçmişi ──────────────────────────────────────────────────────────
# Hoşgeldin mesajı (sohbet boşsa)
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Hoş Geldiniz!</h3>
        <p>
            Ben monitör uzmanı yapay zeka asistanınızım.
            Veritabanında <strong>4.400+</strong> monitör hakkında bilgi bulunuyor.
            Monitör seçimi, karşılaştırma ve teknik özellikler hakkında bana sorabilirsiniz.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Öneri soruları
    example_questions = [
        "🎮 En iyi OLED gaming monitör hangisi?",
        "💼 Ofis için 27 inç 4K monitör öner",
        "🔍 240Hz ve üzeri monitörler",
        "📊 Samsung vs LG OLED karşılaştır"
    ]

    # Butonları merkeze hizalamak için boş kolonlar kullanılabilir veya flex yapısı simüle edilebilir
    # Burada 4 butonu yan yana diziyoruz
    cols = st.columns(len(example_questions))
    for i, question in enumerate(example_questions):
        with cols[i]:
            if st.button(question, key=f"q_suggest_{i}", use_container_width=True):
                st.session_state.suggestion_clicked = question
                st.rerun()

# Mevcut mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# ─── Chat Input ──────────────────────────────────────────────────────────────
prompt = st.chat_input("Monitör hakkında bir soru sorun...", disabled=not st.session_state.vector_db_ready)

# Öneri tıklandıysa prompt'u güncelle
if st.session_state.suggestion_clicked:
    prompt = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None

if prompt:
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Asistan yanıtı (streaming)
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Chat geçmişini hazırla (sadece son birkaç mesaj)
            chat_history = []
            for msg in st.session_state.messages[:-1]:  # Son mesaj hariç
                chat_history.append({"role": msg["role"], "content": msg["content"]})

            # Streaming yanıt
            for chunk in generate_response_stream(prompt, chat_history):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ Bir hata oluştu: {str(e)}"
            message_placeholder.markdown(full_response)

    # Asistan yanıtını kaydet
    st.session_state.messages.append({"role": "assistant", "content": full_response})
