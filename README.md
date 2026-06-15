# 🖥️ Monitor Assistant LLM

## 📌 Proje Hakkında

Bu proje, monitörler hakkında kullanıcıların doğal dilde soru sorabilmesini sağlayan RAG (Retrieval-Augmented Generation) tabanlı bir LLM sistemidir.

Sistem kapsamında monitör verileri web scraping yöntemleri kullanılarak hedef web sitesinden elde edilmiş, daha sonra embedding işlemleri uygulanarak vektör veritabanına kaydedilmiştir. Kullanıcıdan gelen sorular, semantik arama yardımıyla ilgili monitör verileriyle eşleştirilmekte ve büyük dil modeli tarafından anlamlı cevaplar üretilmektedir.

## 🚀 Kullanılan Teknolojiler

- Python
- Web Scraping (BeautifulSoup, Requests)
- Ollama
- Qwen 2.5
- BGE-M3 Embedding Model
- ChromaDB
- RAG (Retrieval-Augmented Generation)

## 🏗️ Sistem Mimarisi

1. Monitör verileri web scraping ile toplanır.
2. Toplanan veriler temizlenerek işlenir.
3. BGE-M3 modeli kullanılarak embedding'ler oluşturulur.
4. Embedding verileri ChromaDB içerisine kaydedilir.
5. Kullanıcının sorusu embedding'e dönüştürülür.
6. ChromaDB üzerinden en alakalı veriler getirilir.
7. Qwen 2.5 modeli bağlamı kullanarak yanıt üretir.

## ✨ Özellikler

- Monitörler hakkında doğal dilde soru-cevap
- Web scraping ile otomatik veri toplama
- Semantik arama desteği
- ChromaDB tabanlı vektör veritabanı
- Yerel ortamda çalışan Ollama entegrasyonu
- RAG mimarisi ile daha doğru cevaplar
- Monitör özellikleri ve karşılaştırmaları hakkında bilgi sunma

## 💬 Örnek Sorular

- Oyun için en uygun monitör hangisidir?
- 144 Hz ve üzeri monitörleri listele.
- IPS panel kullanan monitörler nelerdir?
- 27 inç monitör önerileri nelerdir?
- Bu monitörün avantajları ve dezavantajları nelerdir?
- Fiyat/performans açısından hangi monitör daha iyidir?

## ⚙️ Kurulum

Projeyi klonlayın:

```bash
git clone <repo-url>
cd monitor-assistant
```

Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

Ollama servisini başlatın:

```bash
ollama run qwen2.5
```

Verileri kazıyın:

```bash
python scraper.py
```

Embedding oluşturun ve veritabanına kaydedin:

```bash
python embedding.py
```

Uygulamayı çalıştırın:

```bash
python app.py
```

## 🎯 Projenin Amacı

Bu proje, web scraping, embedding modelleri, vektör veritabanları ve büyük dil modellerinin birlikte kullanıldığı bir RAG sisteminin geliştirilmesini amaçlamaktadır. Kullanıcıların monitörler hakkında hızlı, doğru ve bağlama uygun bilgilere erişebilmesi hedeflenmiştir.

## 📚 Öğrenilen Konular

- Web Scraping
- Veri Ön İşleme
- Embedding Modelleri
- Vektör Veritabanları
- ChromaDB
- Ollama Kullanımı
- Qwen 2.5 Entegrasyonu
- Retrieval-Augmented Generation (RAG)
- LLM Tabanlı Soru-Cevap Sistemleri

## 👨‍💻 Geliştirici

Bu proje eğitim ve araştırma amacıyla geliştirilmiştir.
