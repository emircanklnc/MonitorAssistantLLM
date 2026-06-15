Monitor Assistant LLM
Proje Hakkında

Bu proje, monitörler hakkında kullanıcıların doğal dilde soru sorabilmesini sağlayan bir LLM (Large Language Model) destekli soru-cevap sistemidir.

Sistem, monitör verilerini çeşitli web sitelerinden web scraping yöntemleri ile toplayarak yapılandırılmış hale getirir. Toplanan veriler embedding işleminden geçirilerek vektör veritabanında saklanır ve kullanıcı sorgularına en alakalı sonuçlar Retrieval-Augmented Generation (RAG) yaklaşımı ile sunulur.

Kullanılan Teknolojiler
Python
Web Scraping
Ollama
Qwen 2.5
BGE-M3 Embedding Model
ChromaDB
RAG (Retrieval-Augmented Generation)
Sistem Mimarisi
Monitör bilgileri web scraping yöntemi ile hedef web sitesinden çekilir.
Elde edilen veriler temizlenir ve işlenir.
Metinler BGE-M3 embedding modeli kullanılarak vektörlere dönüştürülür.
Oluşturulan embeddingler ChromaDB içerisinde saklanır.
Kullanıcıdan gelen sorular embeddinge dönüştürülür.
ChromaDB üzerinden en alakalı monitör verileri getirilir.
Qwen 2.5 modeli, getirilen bağlamı kullanarak kullanıcıya anlamlı cevaplar üretir.
Özellikler
Monitörler hakkında doğal dilde soru sorma
Web scraping ile otomatik veri toplama
Vektör tabanlı semantik arama
RAG destekli cevap üretimi
Yerel ortamda çalışan Ollama entegrasyonu
Hızlı ve ölçeklenebilir veri erişimi
Örnek Sorular
Oyun için en uygun 27 inç monitör hangisi?
144 Hz ve üzeri monitörleri listele.
IPS panel kullanan monitörler nelerdir?
5000 TL altındaki monitör seçenekleri nelerdir?
Bu monitörün avantajları ve dezavantajları nelerdir?
Kurulum
git clone <repo-url>
cd monitor-assistant

Gerekli bağımlılıkları yükleyin:

pip install -r requirements.txt

Ollama'yı çalıştırın:

ollama run qwen2.5

Verileri kazıyın:

python scraper.py

Embedding oluşturun:

python embedding.py

Uygulamayı başlatın:

python app.py
Amaç

Bu proje, web scraping, embedding modelleri, vektör veritabanları ve büyük dil modellerinin birlikte kullanıldığı uçtan uca bir RAG sisteminin geliştirilmesini amaçlamaktadır. Kullanıcıların monitör ürünleri hakkında hızlı, doğru ve bağlama uygun bilgi alabilmesi hedeflenmiştir.

Geliştiriciler

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.
