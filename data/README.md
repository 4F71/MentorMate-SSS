# Proje Veri Setleri Açıklaması
 

Merhaba! Bu repo, bir bootcamp'in sıkça sorulan sorularını yanıtlamak üzere geliştirdiğim RAG (Retrieval-Augmented Generation) tabanlı **MentorMate Chatbot** projesinin kodlarını ve hikayesini içeriyor. Bu proje, basit bir SSS botu fikrinden çok daha fazlası oldu; karşılaştığım her zorluk, beni daha profesyonel ve daha dayanıklı çözümler üretmeye iten bir öğrenme serüvenine dönüştü.

Bu `README` dosyası, sadece projenin ne yaptığını değil, aynı zamanda bu noktaya nasıl geldiğimi, hangi engelleri aştığımı ve bu süreçte neler öğrendiğimi anlatıyor.

## Projenin Son Hali: MentorMate Nasıl Çalışıyor?

MentorMate, bir dizi modern yapay zeka aracını ve tekniğini bir araya getiren sağlam bir mimariye sahip:

* **Beyin (LLM):** Kullanıcıyla sohbet eden ve cevapları üreten zeki beyin olarak Google'ın `gemini-2.0-flash` modelini kullanıyorum.
* **Hafıza (Vektör Veritabanı):** Tüm SSS verilerini, yerel olarak çalışan `ChromaDB` veritabanında saklıyorum.
* **Kütüphaneci (Embedding Modeli):** Metinleri yapay zekanın anlayacağı vektörlere dönüştürmek için Hugging Face'in `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modelini **yerel olarak** çalıştırıyorum. Bu sayede API limitlerine takılmadan, tamamen bağımsız bir altyapı kurdum.
* **Akıllı Arama (Retrieval):** Kullanıcının sorduğu bir soruyu birden fazla farklı açıdan ele alıp veritabanında arama yapan `MultiQueryRetriever + MMR (Maximum Marginal Relevance)` tekniğini kullanarak cevap isabet oranını en üst seviyeye çıkardım.
* **Yüz (Arayüz):** Tüm bu güçlü altyapıyı, `app.py` dosyası içinde çalışan interaktif ve kullanıcı dostu bir `Streamlit` web uygulaması ile sunuyorum.

## Veri İşleme Akışı: Ham Veriden Zekaya

Chatbot'un "beyni" ne kadar zekiyse, "hafızası" da o kadar kaliteli olmalıdır. Bu projenin temelini oluşturan veri, aşağıdaki adımlardan geçerek son halini aldı:
