import streamlit as st
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Ortam değişkenlerini (şifreleri) hafızaya yükle
load_dotenv()

# 2. Streamlit Arayüz Ayarları
st.set_page_config(
    page_title="Letterboxd Akıllı Öneri Sistemi", 
    page_icon="🎬", 
    layout="centered"
)

st.title("🎬 Bulut Tabanlı Akıllı Film Öneri Sistemi")
st.markdown("**AWS RDS (PostgreSQL)** üzerindeki veri kümesi ve **TF-IDF + Kosinüs Benzerliği** algoritması kullanan içerik tabanlı öneri motoru.")
st.write("---")

# 3. AWS RDS Veri Çekme Katmanı
@st.cache_data(show_spinner="AWS Bulut Veritabanından 25.000 Film Çekiliyor... Lütfen Bekleyin.")
def load_data_from_aws():
    db_config = {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": os.getenv("DB_PORT", "5432")
    }
    try:
        conn = psycopg2.connect(**db_config)
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
        df = pd.read_sql_query(query, conn)
        conn.close() 
        
        df['metadata'] = df['metadata'].fillna('')
        df['title_lower'] = df['title'].str.lower()
        return df
    except Exception as e:
        st.error(f"⚠️ AWS Bağlantı Hatası! Veritabanının 'Available' (Açık) modda olduğundan emin olun.")
        return None

# 4. Yapay Zeka Hesaplama Motoru
@st.cache_resource
def calculate_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Veriyi ve matrisi yükle
df = load_data_from_aws()

if df is not None:
    cosine_sim = calculate_similarity_matrix(df)
    st.success("✅ AWS Veri Akışı ve Yapay Zeka Motoru Stabil!")

    # 5. Kullanıcı Seçim Alanı
    movie_list = df['title'].values
    selected_movie = st.selectbox(
        "Öneri almak istediğiniz filmi seçin veya yazın:",
        movie_list,
        index=None,
        placeholder="Bir film başlığı giriniz..."
    )

    if selected_movie:
        search_title = selected_movie.lower()
        matched = df[df['title_lower'] == search_title]
        
        if not matched.empty:
            idx = matched.index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]
            movie_indices = [i[0] for i in sim_scores]
            
            st.write("---")
            st.subheader(f"🍿 '{selected_movie}' İçin Film Önerileri")
            
            for i, row in enumerate(df.iloc[movie_indices].itertuples(), 1):
                clean_metadata = ", ".join(row.metadata.split())
                with st.container():
                    st.markdown(f"### {i}. {row.title}")
                    st.caption(f"**Özellik Havuzu (Genres + Themes):** {clean_metadata}")
                    st.write("")