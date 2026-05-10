"""
RAG (Retrieval-Augmented Generation) Sistemi
Monitör verileri üzerinde Ollama + Qwen 2.5 ile çalışan chatbot altyapısı.
"""

import os
import shutil
import pandas as pd
import requests
import json
import chromadb

# ─── Konfigürasyon ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitör", "monitorler.csv")
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "monitorler"
TOP_K = 8  # RAG sorgusunda getirilecek en alakalı belge sayısı


# ─── Ollama Bağlantı Kontrolü ────────────────────────────────────────────────
def check_ollama_connection():
    """Ollama sunucusunun çalışıp çalışmadığını kontrol eder."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def check_model_available(model_name):
    """Belirtilen modelin Ollama'da yüklü olup olmadığını kontrol eder."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model_name in model.get("name", ""):
                    return True
        return False
    except requests.ConnectionError:
        return False


def get_system_status():
    """Sistem durumunu kontrol eder ve bir sözlük döndürür."""
    status = {
        "ollama_connected": check_ollama_connection(),
        "embedding_model": check_model_available(EMBEDDING_MODEL),
        "llm_model": check_model_available(LLM_MODEL),
        "csv_exists": os.path.exists(CSV_PATH),
        "vector_db_exists": os.path.exists(CHROMA_DB_PATH),
    }
    status["all_ready"] = all(status.values())
    return status


# ─── CSV Veri Yükleme ─────────────────────────────────────────────────────────
def load_csv_data():
    """
    CSV dosyasını okur, her satırı bir döküman (chunk) olarak hazırlar.
    'Sablon' sütunu ana metin, diğer sütunlar metadata olarak kullanılır.
    """
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8", dtype=str)
    df = df.fillna("")

    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Sablon sütunu zaten Türkçe doğal dilde hazır açıklama içeriyor
        text = row.get("Sablon", "")
        if not text.strip():
            # Sablon yoksa diğer sütunlardan birleştir
            text = (
                f"{row.get('UrunAdi', '')} monitör. "
                f"Çözünürlük: {row.get('CozunurlukBilgisi', '')}. "
                f"Fiyat: {row.get('Fiyat', 'belirtilmemiş')}. "
                f"Ekran boyutu: {row.get('EkranBoyutu', '')}. "
                f"Yenileme hızı: {row.get('YenilemeHizi', '')}. "
                f"Ekran teknolojisi: {row.get('EkranTeknolojisi', '')}. "
                f"Teknik puan: {row.get('TeknikPuan', '')}."
            )

        metadata = {
            "urun_adi": str(row.get("UrunAdi", "")),
            "cozunurluk": str(row.get("CozunurlukBilgisi", "")),
            "fiyat": str(row.get("Fiyat", "")),
            "ekran_boyutu": str(row.get("EkranBoyutu", "")),
            "yenileme_hizi": str(row.get("YenilemeHizi", "")),
            "ekran_teknolojisi": str(row.get("EkranTeknolojisi", "")),
            "teknik_puan": str(row.get("TeknikPuan", "")),
        }

        documents.append(text)
        metadatas.append(metadata)
        ids.append(f"monitor_{idx}")

    return documents, metadatas, ids


# ─── Embedding ────────────────────────────────────────────────────────────────
def get_embeddings(texts, model=EMBEDDING_MODEL):
    """
    Ollama API üzerinden metin listesi için embedding vektörleri oluşturur.
    Toplu (batch) olarak gönderir — GPU paralel işler, çok daha hızlı.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=600,
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("embeddings", [])
    else:
        raise Exception(f"Embedding hatası: {response.status_code} - {response.text}")


def get_single_embedding(text, model=EMBEDDING_MODEL):
    """Tek bir metin için embedding vektörü oluşturur."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": text},
        timeout=60,
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("embeddings", [[]])[0]
    else:
        raise Exception(f"Embedding hatası: {response.status_code} - {response.text}")


# ─── Vektör Veritabanı (ChromaDB) ────────────────────────────────────────────
def get_chroma_client():
    """ChromaDB istemcisi oluşturur (persistent storage)."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client


def build_vector_store(progress_callback=None):
    """
    CSV verisini yükler, embedding oluşturur ve ChromaDB'ye kaydeder.
    progress_callback: İlerleme durumunu bildirmek için callback fonksiyonu.
    """
    # Veriyi yükle
    documents, metadatas, ids = load_csv_data()
    total = len(documents)

    if progress_callback:
        progress_callback(0, total, "CSV verisi yüklendi...")

    # Eski veritabanını tamamen temizle (stale cache sorununu önler)
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)

    # Yeni ChromaDB istemcisi ve koleksiyon
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch halinde embedding oluştur ve ekle (GPU batch processing)
    batch_size = 200
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch_docs = documents[i:batch_end]
        batch_metas = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]

        # Embedding oluştur
        batch_embeddings = get_embeddings(batch_docs)

        # ChromaDB'ye ekle
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
            ids=batch_ids,
        )

        if progress_callback:
            progress_callback(batch_end, total, f"{batch_end}/{total} döküman işlendi...")

    return total


def get_collection():
    """Mevcut ChromaDB koleksiyonunu döndürür."""
    if not os.path.exists(CHROMA_DB_PATH):
        return None
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        return collection
    except Exception:
        return None


def vector_store_exists():
    """Vektör veritabanının dolu olup olmadığını kontrol eder."""
    if not os.path.exists(CHROMA_DB_PATH):
        return False
    try:
        collection = get_collection()
        if collection is None:
            return False
        return collection.count() > 0
    except Exception:
        return False


# ─── RAG Sorgu ────────────────────────────────────────────────────────────────
def query_similar_documents(query_text, top_k=TOP_K):
    """
    Kullanıcı sorgusuna en benzer dökümanları ChromaDB'den getirir.
    """
    collection = get_collection()
    if collection is None:
        return [], []

    # Sorgu için embedding oluştur
    query_embedding = get_single_embedding(query_text)

    # Benzer dökümanları getir
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return documents, metadatas


# ─── LLM Yanıt Üretme ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Sen bir monitör uzmanısın. Kullanıcıların monitör seçimi ve monitör özellikleri hakkındaki sorularına yardımcı oluyorsun.

Sana verilen BAĞLAM bilgilerini kullanarak soruları yanıtla. Bağlamda bulunan monitör verilerini temel alarak doğru ve detaylı bilgi ver.

Kurallar:
1. Sadece bağlamda bulunan verilere dayanarak yanıt ver.
2. Eğer bağlamda yeterli bilgi yoksa, bunu belirt.
3. Fiyatları, teknik özellikleri ve puanları doğru şekilde aktar.
4. Karşılaştırma istendiğinde tablo formatı kullan.
5. Türkçe yanıt ver.
6. Önerilerde bulunurken teknik puanı, ekran teknolojisini ve kullanım amacını göz önünde bulundur.
7. Yanıtlarını düzenli ve okunabilir bir formatta sun."""


def generate_response_stream(user_query, chat_history=None):
    """
    RAG pipeline: Sorguyu embedding'e çevir → benzer dökümanları getir → LLM'e gönder.
    Streaming yanıt döndürür (generator).
    """
    # 1. Benzer dökümanları getir
    documents, metadatas = query_similar_documents(user_query)

    # 2. Bağlam oluştur
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        context_parts.append(f"[Monitör {i}] {doc}")

    context = "\n\n".join(context_parts)

    # 3. Prompt oluştur
    user_message = f"""BAĞLAM (Veritabanından getirilen monitör bilgileri):
{context}

KULLANICI SORUSU:
{user_query}

Lütfen yukarıdaki bağlam bilgilerini kullanarak kullanıcının sorusunu yanıtla."""

    # 4. Mesaj geçmişini hazırla
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:  # Son 3 sohbet çifti
            messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    # 5. Ollama API'ye streaming istek gönder
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 2048,
            },
        },
        stream=True,
        timeout=120,
    )

    if response.status_code != 200:
        yield f"Hata: LLM yanıt veremedi (HTTP {response.status_code})"
        return

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
