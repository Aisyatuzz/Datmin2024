import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Fungsi untuk mengunduh file dan memuat dengan pickle
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Gagal mengunduh file dari URL: {url}")
        return None

# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Analisis Sentimen SpotifyWrapped 2024")

    # Load model dan vectorizer dari URL
    model_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/rf_model.pkl"
    vectorizer_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/vectorizer.pkl"

    model = load_model_from_url(model_url)
    vectorizer = load_model_from_url(vectorizer_url)

    # Pastikan model dan vectorizer berhasil di-load
    if model and vectorizer:
        # Bagian 1: Prediksi Sentimen berdasarkan Input Teks
        st.header("Prediksi Sentimen Berdasarkan Input Teks")
        user_input = st.text_area("Masukkan teks untuk prediksi sentimen:")
        if st.button("Prediksi Sentimen Teks"):
            if user_input.strip():
                # Transformasi teks menggunakan vectorizer
                input_vectorized = vectorizer.transform([user_input])
                
                # Prediksi sentimen
                sentiment_prediction = model.predict(input_vectorized)[0]
                
                # Tampilkan hasil prediksi
                st.success(f"Hasil Prediksi: **{sentiment_prediction}**")
            else:
                st.warning("Masukkan teks terlebih dahulu.")

        # Bagian 2: Prediksi Sentimen dari File CSV
        st.header("Prediksi Sentimen dari File CSV")
        uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
        if uploaded_file is not None:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.write("Data yang diunggah:")
            st.write(data)

            # Validasi kolom 'stemming_data'
            if 'stemming_data' in data.columns:
                # Transformasi data menggunakan vectorizer
                X_test = vectorizer.transform(data['stemming_data'])

                # Prediksi Sentimen
                if st.button("Prediksi Sentimen dari CSV"):
                    predictions = model.predict(X_test)
                    
                    # Tambahkan hasil prediksi ke data
                    data['Predicted Sentiment'] = predictions
                    st.write("Hasil Prediksi Sentimen:")
                    st.write(data[['stemming_data', 'Predicted Sentiment']])

                    # Visualisasi distribusi sentimen
                    sentiment_counts = data['Predicted Sentiment'].value_counts()
                    fig_bar = px.bar(
                        sentiment_counts,
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        labels={'x': 'Sentimen', 'y': 'Jumlah'},
                        title="Distribusi Sentimen"
                    )
                    st.plotly_chart(fig_bar)

                    # Tombol untuk mengunduh hasil prediksi
                    st.download_button(
                        label="Download Hasil Prediksi",
                        data=data.to_csv(index=False),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Kolom 'stemming_data' tidak ditemukan dalam file yang diunggah.")

if __name__ == '__main__':
    main()
