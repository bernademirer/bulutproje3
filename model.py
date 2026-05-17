import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys


def create_aws_recommendation_system():
    # -------------------------------------------------------------------------
    # 1. AWS RDS POSTGRESQL BAĞLANTI AYARLARI
    # -------------------------------------------------------------------------
    # Terminal çıktındaki gerçek adresin ve veritabanı adın tanımlanmıştır.
    db_config = {
        "host": "letterboxd-db.cm5wqgog6a02.us-east-1.rds.amazonaws.com", 
        "database": "letterboxd_db",
        "user": "postgres",
        "password": "elateocanber",  # <-- Buraya AWS veritabanı şifreni yaz
        "port": "5432"
    }
    
    print("AWS RDS Bulut Veritabanına bağlanılıyor...")
    try:
        conn = psycopg2.connect(**db_config)
        print("AWS Bulut veritabanı bağlantısı başarıyla sağlandı!")
    except Exception as e:
        print(f"Bağlantı Hatası! AWS Güvenlik Grubu (Security Group) izinlerini kontrol edin.\nDetay: {e}")
        sys.exit()

    # -------------------------------------------------------------------------
    # 2. VERİDEN BİLGİ ÇIKARMA (ÖZELLİK MÜHENDİSLİĞİ - JOIN VE AGGREGATION)
    # -------------------------------------------------------------------------
    # DISTINCT kullanarak türlerin ve temaların mükerrer (çoğul) gelmesini engelliyoruz.
    # COALESCE ile boş (NULL) gelen verileri temiz metne dönüştürüyoruz.
    query = """
        SELECT m.id, m.name as title, 
               TRIM(COALESCE(STRING_AGG(DISTINCT g.name, ' '), '') || ' ' || COALESCE(STRING_AGG(DISTINCT t.name, ' '), '')) as metadata
        FROM public.movies m
        LEFT JOIN public.genres g ON m.id = g.movie_id
        LEFT JOIN public.themes t ON m.id = t.movie_id
        GROUP BY m.id, m.name
        ORDER BY m.id ASC
        LIMIT 25000;
    """
    
    print("AWS üzerindeki veri kümesi (Movies + Genres + Themes) SQL ile çekiliyor...")
    df = pd.read_sql_query(query, conn)
    conn.close() 
    print(f"Veriler başarıyla DataFrame'e aktarıldı. Toplam film sayısı: {len(df)}")
    
    # -------------------------------------------------------------------------
    # 3. DOĞAL DİL İŞLEME (NLP) VE VEKTÖRLEŞTİRME
    # -------------------------------------------------------------------------
    # Metadatası tamamen boş kalmış filmleri boş string ile dolduruyoruz.
    df['metadata'] = df['metadata'].fillna('')
    
    print("TF-IDF algoritması ile metinsel veriler matematiksel matrise dönüştürülüyor...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    
    # -------------------------------------------------------------------------
    # 4. MAKİNE ÖĞRENMESİ ALGORİTMASI (KOSİNÜS BENZERLİĞİ)
    # -------------------------------------------------------------------------
    print("Kosinüs Benzerliği matrisi hesaplanıyor...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Arama yaparken küçük-büyük harf duyarlılığını ortadan kaldırıyoruz
    df['title_lower'] = df['title'].str.lower()
    
    # -------------------------------------------------------------------------
    # 5. AKILLI FİLM ÖNERİ MEKANİZMASI
    # -------------------------------------------------------------------------
    def get_recommendations(movie_title, top_n=5):
        search_title = movie_title.lower()
        matched = df[df['title_lower'] == search_title]
        
        if matched.empty:
            print(f"\n Hata: '{movie_title}' filmi bulut veri setinde bulunamadı.")
            return
        
        idx = matched.index[0]
        actual_title = matched.iloc[0]['title']

        # Benzerlik skorlarını listele ve en yüksekten düşüğe doğru sırala
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Kendisi hariç en benzer top_n filmi seç
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]

        print(f"\n--- '{actual_title}' İçin AWS & Yapay Zeka Önerileri ---")
        for i, row in enumerate(df.iloc[movie_indices].itertuples(), 1):
            # Temiz bir çıktı için fazla boşlukları temizleyerek ekrana yazdırıyoruz
            clean_metadata = ", ".join(row.metadata.split())
            print(f"{i}. {row.title} ")

    return get_recommendations

# -------------------------------------------------------------------------
# MODELİ ÇALIŞTIRMA VE TEST ALANI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    recommend = create_aws_recommendation_system()
    
    # İstediğin herhangi bir popüler filmle sistemi test edebilirsin:
    recommend("The Hangover")