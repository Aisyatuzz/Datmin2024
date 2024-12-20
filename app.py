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
    st.title("Analisis Sentiment SpotifyWrapped 2024")

    # Bagian untuk upload file
    st.subheader("Prediksi Sentimen Berdasarkan File CSV")
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Load model dan vectorizer dari URL
        model_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/rf_model.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        # Pastikan model dan vectorizer berhasil di-load
        if model and vectorizer:
            # Validasi kolom 'stemming_data'
            if 'stemming_data' in data.columns:
                # Transformasi data menggunakan vectorizer
                X_test = vectorizer.transform(data['stemming_data'])

                # Prediksi Sentimen
                if st.button("Prediksi Sentiment CSV", key="predict_csv_button"):
                    # Prediksi dengan model yang sudah dilatih
                    predictions = model.predict(X_test)

                    # Tambahkan hasil prediksi ke data
                    data['Predicted Sentiment'] = predictions

                    # Simpan hasil prediksi di session_state
                    st.session_state['csv_predictions'] = data

                    st.write("Hasil Prediksi Sentiment:")
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

                    # Evaluasi akurasi jika tersedia label 'sentiment'
                    if 'sentiment' in data.columns:
                        accuracy = accuracy_score(data['sentiment'], predictions)
                        report_dict = classification_report(data['sentiment'], predictions, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.success(f"Akurasi Model: {accuracy:.2%}")
                        st.write("Laporan Klasifikasi:")
                        st.table(report_df)

                    # Tombol untuk mengunduh hasil prediksi
                    st.download_button(
                        label="Download Hasil Prediksi",
                        data=data.to_csv(index=False),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )

    # Input untuk prediksi kata/kalimat
    st.subheader("Prediksi Sentimen Berdasarkan Input Teks")
    user_input = st.text_input("Masukkan teks Anda di sini:")
    if st.button("Prediksi Sentiment Input Teks", key="predict_text_button"):  # Beri key unik
        if user_input and model and vectorizer:
            # Transformasi teks menggunakan vectorizer
            input_vectorized = vectorizer.transform([user_input])
        
            # Prediksi sentimen
            sentiment_prediction = model.predict(input_vectorized)[0]
            st.write(f"Prediksi Sentimen untuk teks: **{sentiment_prediction}**")

if __name__ == '__main__':
    main()
