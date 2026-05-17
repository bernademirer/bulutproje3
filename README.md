# 3522 Bulut Bilişim Dersi - Proje 3
## Bulut Tabanlı Akıllı Film Öneri Sistemi ve Makine Öğrenmesi Uygulaması

### Proje Amacı ve Kapsamı
Bu projenin amacı; büyük ölçekli film veri kümeleri (Letterboxd Dataset) üzerinden anlamsal özellikler çıkararak kullanıcılara kişiselleştirilmiş akıllı film önerileri sunan hibrit bir bulut sistemidir. Uygulama, filmlerin tür (genres) ve temalarını (themes) doğal dil işleme teknikleriyle analiz ederek içerik tabanlı filtreleme (Content-Based Filtering) yöntemiyle benzerlik skorları üretir.

### Kullanılan Teknolojiler
AWS RDS (PostgreSQL)
Psycopg2
Pandas
Scikit-learn (TF-IDF)
Scikit-learn (Cosine Similarity)
Streamlit
#### **Gün 1 - 16.05.2026: Altyapı Analizi ve Mimari Tasarım**
* Proje reposu oluşturuldu ve Kaggle üzerinden büyük ölçekli Letterboxd veri seti indirildi.
* Lokal veritabanı kurulumunun bilgisayara getireceği yük ve hantallık analiz edilerek, veritabanı katmanının tamamen **AWS RDS** üzerine taşınmasına (Cloud Migration) karar verildi.
* Veri kümesinde yer alan büyük boyutlu SQL dosyalarını projenin gereksinimlerine göre elendi. Makine öğrenmesi modelini besleyecek en verimli tabloların filmler, türler ve temalar olduğu saptandı; aktör ve ekip dosyaları sisteme dahil edilmeyerek optimizasyon sağlandı.
*AWSde veritabanı oluşturuldu ve indirdiğimiz veri setini AWS içinde yapılandırarak veri aktarımı tamamlandı.

#### ** Gün 2 17.05.2026 Model Doğrulaması, Streamlit Entegrasyonu
*`model.py` scripti yazıldı. AWS RDS PostgreSQL veritabanına bağlanılarak `LEFT JOIN`, `DISTINCT` ve `COALESCE` optimizasyonlarıyla harmanlanmış SQL sorgusuyla 25.000 satırlık veri seti çekildi.
*Projeyi komut satırından kurtarmak ve son kullanıcıya sunulabilir hale getirmek için `app.py` dosyası üzerinden **Streamlit** entegrasyonu yapıldı. Arama çubuğu ve akıllı film öneri kartları tasarlandı.
*Arayüzün her etkileşimde buluta gidip 25.000 satırı tekrar çekmesini engellemek amacıyla Streamlit'in `@st.cache_data` (veri önbellekleme) ve `@st.cache_resource` mimarileri kullanılarak sistem hızı optimize edildi.
 
 





