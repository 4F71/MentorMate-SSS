# Proje Veri Setleri Açıklaması

Bu klasör, MentorMate SSS Botu projesinin veri hazırlama sürecinde kullanılan kaynak dosyaları içermektedir. Veri akışı aşağıdaki gibidir:

### 1. `zulip_data.txt`

- **Amaç:** Ham Veri Kaynağı
- **Açıklama:** Bu dosya, projenin başlangıç noktasını oluşturan, Zulip platformundan alınmış en ham ve işlenmemiş Soru-Cevap metinlerini içerir. Bootcamp başlangıcında en sık sorulan sorular bulunur. Formatı basit bir metin dosyasıdır.

### 2. `sss_dataset_augmented.json` ve `sss_dataset_heavily_augmented.json` (v2 dahil)

- **Amaç:** Yapılandırılmış ve Zenginleştirilmiş Veri
- **Açıklama:** Ham `zulip_data.txt` dosyasındaki veriler işlenerek bu JSON dosyaları oluşturulmuştur. Bu dosyalarda, her bir ana cevap için birden çok soru varyasyonu (`all_questions`) gruplandırılmıştır. `heavily` ve `v2` sürümleri, daha fazla soru varyasyonu eklenmiş daha gelişmiş versiyonları temsil eder.

### Not: Nihai Çalışma Dosyası

Bu klasördeki dosyalar, projenin ham ve ara veri kaynaklarıdır. Botun veritabanını oluşturmak için kullanılan nihai, temizlenmiş ve düzleştirilmiş `.jsonl` formatındaki dosya, projenin `output` klasöründe yer almaktadır.