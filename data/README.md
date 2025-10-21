# MentorMate SSS - Veri Seti Dokümantasyonu

Bu klasör, **Akbank GenAI Bootcamp - MentorMate SSS Chatbot** projesinin Bilgi Tabanı'nı (Knowledge Base) oluşturmak için kullanılan tüm veri setlerini içerir.

Buradaki dosyalar, ham veriden başlayarak, yapay zeka (Google Gemini) ile zenginleştirilmiş (augmented) ve sıfırdan üretilmiş (generated) Soru-Cevap (Q&A) çiftlerine kadar tüm veri hazırlık sürecini göstermektedir.

Nihai amaç, RAG (Retrieval-Augmented Generation) modelinin vektör veritabanını (`ChromaDB`) beslemek için zengin ve çeşitli bir SSS veri seti oluşturmaktır.

## Dosya Açıklamaları

Bu klasördeki her dosya, veri işleme hattının farklı bir aşamasını temsil eder:

### 1. Ham Veri

* `zulip_data.txt`
    * **Açıklama:** Veri toplama aşamasının ham çıktısıdır. Muhtemelen Bootcamp'in Zulip/Slack/Discord kanalından dışa aktarılan, SSS oluşturmak için temel kaynak olarak kullanılan filtrelenmemiş sohbet kayıtlarını içerir.

### 2. Zenginleştirilmiş (Augmented) Veri Setleri

Bu dosyalar, temel bir SSS setinin (muhtemelen `zulip_data.txt`'den çıkarılan) alınıp, LLM (Gemini) kullanılarak varyasyonlarının üretildiği dosyalardır. Amaç, botun aynı soruyu farklı şekillerde soran kullanıcıları da anlamasını sağlamaktır.

* `sss_dataset_augmented.json`
    * **Açıklama:** Temel SSS veri setinin ilk zenginleştirme (augmentation) işleminden geçmiş halidir.
* `sss_dataset_heavily_augmented.json`
* `sss_dataset_heavily_augmented_v2.json`
    * **Açıklama:** Temel veri setinin, daha fazla çeşitlilik ve farklı soru kalıpları oluşturmak amacıyla "yoğun" bir zenginleştirme (heavy augmentation) işleminden geçirilmiş versiyonlarıdır.

### 3. Yapay Zeka Tarafından Üretilen (Generated) Veri

* `generated_data_google.jsonl`
    * **Açıklama:** Google'ın Generative AI modeli (Gemini) tarafından sıfırdan üretilmiş Soru-Cevap (Q&A) çiftlerini içerir. Bu, mevcut SSS'lerin ötesinde, modelin konu hakkında "düşünerek" yeni sorular üretmesini sağlayarak bilgi tabanını genişletir. `.jsonl` formatındadır (her satır bir JSON nesnesidir).

### 4. Nihai Veri Seti

* `enriched_dataset.jsonl`
    * **Açıklama:** **Bu, vektör veritabanını beslemek için kullanılan NİHAİ dosyadır.** Yukarıdaki tüm kaynakların (`augmented`, `generated` ve muhtemelen `zulip_data.txt`'den çıkarılan temel SSS'ler) birleştirilmiş, temizlenmiş ve RAG sistemi için optimize edilmiş son halidir. `setup_database.py` script'i bu dosyayı okuyarak `ChromaDB` koleksiyonunu oluşturur.

---

## Veri İşleme Akışı (Workflow)

1.  **Toplama:** `zulip_data.txt` içerisinden manuel olarak temel SSS'ler ayıklandı.
2.  **Zenginleştirme (Augmentation):** Temel SSS'ler, farklı soru varyasyonları oluşturması için bir LLM'e (Gemini) beslendi (`sss_..._augmented.json` dosyaları).
3.  **Üretme (Generation):** LLM'den (Gemini) konuyla ilgili sıfırdan yeni Soru-Cevap çiftleri üretmesi istendi (`generated_data_google.jsonl`).
4.  **Birleştirme ve Temizleme:** Tüm bu veri setleri bir araya getirilerek `enriched_dataset.jsonl` oluşturuldu.
5.  **Yükleme:** Bu nihai dosya, `setup_database.py` script'i tarafından okunarak vektörlere dönüştürüldü ve `ChromaDB` veritabanına meta-verileri ile birlikte yüklendi.
