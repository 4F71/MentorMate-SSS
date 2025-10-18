MentorMate SSS Chatbot 🤖
Bu proje, bir bootcamp'in sıkça sorulan sorularına (SSS) yanıt vermek üzere tasarlanmış, RAG (Retrieval-Augmented Generation) mimarisine sahip bir yapay zeka chatbot uygulamasıdır. MentorMate, Streamlit ile geliştirilmiş interaktif bir web arayüzü üzerinden hizmet verir ve iki farklı kişilikte cevaplar üretebilir:

Bootcamp Uzmanı: Kendi vektör veritabanında (ChromaDB) bulduğu kesin bilgilere dayanarak, kaynak belirterek cevap verir.

Genel Yardımcı Asistan: Veritabanında bilgi bulamadığında, bu durumu belirterek genel yapay zeka bilgisiyle kullanıcıya yardımcı olur.

✨ Özellikler
Çift Kişilikli Cevaplama: Sorunun cevabının veritabanında olup olmamasına göre dinamik olarak rol değiştirir.

İnteraktif Web Arayüzü: Streamlit kullanılarak modern ve kullanıcı dostu bir sohbet arayüzü sunar.

Kaynak Gösterme: Veritabanından verilen cevapların sonunda, bilginin hangi dokümandan alındığını belirterek güvenilirliği artırır.

Sohbet Hafızası: Konuşma bağlamını korumak için son birkaç adımı hatırlar.

Akıllı Arama: MultiQueryRetriever kullanarak, kullanıcının sorduğu sorunun farklı varyasyonlarını üreterek veritabanında daha isabetli aramalar yapar.

Yerel Embedding: API maliyetlerinden kaçınmak ve performansı artırmak için Sentence Transformers ile yerel bir embedding modeli kullanır.

🛠️ Kullanılan Teknolojiler
Backend & AI: LangChain, Google Gemini, ChromaDB, Sentence Transformers (Hugging Face)

Frontend: Streamlit

Dil: Python

🚀 Kurulum Adımları
Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1. Gereksinimler
Python 3.9 veya daha üstü

Git

2. Projeyi Klonlama
Terminali açın ve projeyi klonlamak için aşağıdaki komutu çalıştırın:

Bash

git clone https://github.com/4F71/MentorMate-SSS.git
3. Proje Dizinine Gitme
Klonladığınız proje klasörünün içine girin:

Bash

cd MentorMate-SSS
4. API Anahtarı Kurulumu (Çok Önemli!)
Bu projenin Google Gemini modelini kullanabilmesi için bir API anahtarına ihtiyacı vardır. Anahtarınızı güvende tutmak için bir .env dosyası kullanacağız.

Projenin ana dizininde (app.py ile aynı yerde) .env adında yeni bir dosya oluşturun.

Oluşturduğunuz .env dosyasının içine kendi Google API anahtarınızı aşağıdaki formatta yapıştırın:

GOOGLE_API_KEY="AIzaSy...SizinGercekAnahtarınız...xyz"
Not: .gitignore dosyası, .env dosyanızın yanlışlıkla GitHub'a yüklenmesini engelleyecektir. Bu dosyayı asla herkese açık bir alanda paylaşmayın.

5. Sanal Ortam Oluşturma ve Aktif Etme
Kütüphane çakışmalarını önlemek için bir sanal ortam oluşturmak en iyi pratiktir.

Sanal ortamı oluşturun:

Bash

python -m venv .venv
Sanal ortamı aktif hale getirin:

Windows'ta:

Bash

.venv\Scripts\activate
macOS veya Linux'ta:

Bash

source .venv/bin/activate
6. Bağımlılıkları Yükleme
Sanal ortam aktifken, projenin ihtiyaç duyduğu tüm kütüphaneleri requirements.txt dosyası ile tek seferde kurun:

Bash

pip install -r requirements.txt
▶️ Uygulamayı Çalıştırma
Tüm kurulum adımları tamamlandıktan sonra, sanal ortamınızın aktif olduğundan emin olun ve terminale aşağıdaki komutu yazın:

Bash

streamlit run app.py
Bu komut, web tarayıcınızda otomatik olarak yeni bir sekme açacak ve uygulamayı http://localhost:8501 adresinde başlatacaktır. Artık MentorMate ile sohbet etmeye başlayabilirsiniz!

📂 Proje Yapısı
MentorMate-SSS/
│
├── .venv/              # Projenin sanal ortamı
├── chroma_db/          # Vektör veritabanının saklandığı klasör
├── notebooks/          # Geliştirme sürecindeki Jupyter Notebook'lar
├── output/             # Veri işleme çıktılarının bulunduğu klasör
│
├── .env                # (Oluşturulacak) Gizli API anahtarını tutan dosya
├── .gitignore          # Git tarafından yoksayılacak dosyaları (örn: .env) listeler
├── app.py              # Streamlit uygulamasının ana kodu
├── requirements.txt    # Projenin ihtiyaç duyduğu Python kütüphaneleri
└── README.md           # Bu dosya
