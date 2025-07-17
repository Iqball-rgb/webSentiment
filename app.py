import streamlit as st
import pandas as pd
import pickle
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# ===== LOAD MODEL DAN VECTORIZER =====
with open("models/TfidfVectorizer.pkl", "rb") as f:
    vect = pickle.load(f)

with open("models/MultinomialNB_Model.pkl", "rb") as f:
    model_mnb = pickle.load(f)

with open("models/GaussianNB_Model.pkl", "rb") as f:
    model_gnb = pickle.load(f)

# ===== CLEANING FUNCTION =====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== CARI KOLOM TEKS OTOMATIS =====
def get_text_column(df):
    for col in df.columns:
        if re.search(r'(full_text|text|tweet)', col, re.IGNORECASE):
            return col
    return None

# ===== JUDUL =====
st.title("📊 Aplikasi Analisis Sentimen Naive Bayes")
st.markdown("**Model:** MultinomialNB vs GaussianNB")

# ===== UPLOAD CSV =====
st.header("📂 Upload CSV untuk Prediksi")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_col = get_text_column(df)
    
    if not text_col:
        st.error("Tidak ditemukan kolom teks (harus ada kolom seperti 'full_text', 'text', atau 'tweet').")
    else:
        st.write(f"✅ Kolom teks yang digunakan: `{text_col}`")
        st.dataframe(df.head())

        # Preprocessing
        df['clean_text'] = df[text_col].apply(clean_text)

        # Transform TF-IDF
        X_tfidf = vect.transform(df['clean_text'])

        # ===== PREDIKSI DUA MODEL =====
        df['Prediction_MNB'] = model_mnb.predict(X_tfidf)

        # GaussianNB butuh array dense
        X_dense = X_tfidf.toarray()
        df['Prediction_GNB'] = model_gnb.predict(X_dense)

        # ===== TAMPILKAN HASIL =====
        st.subheader("📋 Hasil Prediksi Kedua Model")
        st.dataframe(df[[text_col, 'Prediction_MNB', 'Prediction_GNB']])

        # Download hasil
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")

        # ===== DISTRIBUSI GRAFIK =====
        st.subheader("📊 Distribusi Prediksi Per Model")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**MultinomialNB**")
            counts_mnb = df['Prediction_MNB'].value_counts()
            fig_mnb = px.bar(x=counts_mnb.index, y=counts_mnb.values, title="Distribusi MNB", text=counts_mnb.values)
            st.plotly_chart(fig_mnb, key="mnb_bar")

        with col2:
            st.write("**GaussianNB**")
            counts_gnb = df['Prediction_GNB'].value_counts()
            fig_gnb = px.bar(x=counts_gnb.index, y=counts_gnb.values, title="Distribusi GNB", text=counts_gnb.values)
            st.plotly_chart(fig_gnb, key="gnb_bar")

        # ===== PIE CHART =====
        st.subheader("📈 Pie Chart Perbandingan")
        pie_col1, pie_col2 = st.columns(2)

        with pie_col1:
            fig_pie_mnb = px.pie(names=counts_mnb.index, values=counts_mnb.values, title="Pie MultinomialNB")
            st.plotly_chart(fig_pie_mnb, key="mnb_pie")

        with pie_col2:
            fig_pie_gnb = px.pie(names=counts_gnb.index, values=counts_gnb.values, title="Pie GaussianNB")
            st.plotly_chart(fig_pie_gnb, key="gnb_pie")

        # ===== WORDCLOUD =====
        st.subheader("☁ WordCloud Per Model")
        for model_name, pred_col in [("MultinomialNB", 'Prediction_MNB'), ("GaussianNB", 'Prediction_GNB')]:
            st.write(f"### {model_name}")
            labels = df[pred_col].unique()
            for label in labels:
                subset = df[df[pred_col] == label]
                words = ' '.join(subset['clean_text'])
                wc = WordCloud(width=800, height=400, background_color='white').generate(words)
                st.write(f"**{label}**")
                st.image(wc.to_array())

# ===== PREDIKSI KALIMAT BARU =====
st.header("🔮 Prediksi Kalimat Baru")
user_input = st.text_area("Masukkan kalimat:")
if st.button("Prediksi"):
    if user_input:
        clean_input = clean_text(user_input)
        input_tfidf = vect.transform([clean_input])
        pred_mnb = model_mnb.predict(input_tfidf)[0]
        pred_gnb = model_gnb.predict(input_tfidf.toarray())[0]
        st.success(f"**MultinomialNB:** {pred_mnb} | **GaussianNB:** {pred_gnb}")
