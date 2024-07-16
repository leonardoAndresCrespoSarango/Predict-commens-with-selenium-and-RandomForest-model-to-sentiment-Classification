import os
import random
import time
import joblib
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt
import csv
import logging
from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from tqdm import tqdm
import pandas as pd
import copy
import string
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


####################################################
# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('vader_lexicon')
# Cargar el modelo de lenguaje en español de spaCy
nlp = spacy.load('es_core_news_sm')

stopwords_es = stopwords.words('spanish')
lemmatizer = WordNetLemmatizer()
PUNCT_TO_REMOVE = string.punctuation
analyzer = SentimentIntensityAnalyzer()

# Configuración de logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Definir el diccionario de mapeo de clases
class_mapping = {
    0: "neutral",
    1: "positive",
    2: "negative"
}

@app.route('/')
def index():
    return render_template('index.html')

import time

@app.route('/extract_comments', methods=['POST'])
def extract_comments():
    start_time = time.time()  # Iniciar el temporizador

    youtuber_name = request.form['youtuber_name']
    num_videos = int(request.form['num_videos'])

    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    comments = []
    try:
        for i in range(num_videos):
            driver.get("https://www.youtube.com")
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "search_query"))
            )
            search_box.send_keys(youtuber_name)
            search_box.send_keys(Keys.RETURN)
            time.sleep(5)

            channel_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.ID, "channel-title"))
            )
            for channel in channel_elements:
                if channel.text.strip().lower() == youtuber_name.strip().lower():
                    channel.click()
                    break
            else:
                return jsonify({"error": "No se encontró el canal exacto"})

            time.sleep(5)
            videos_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[@href and contains(@href, '/videos')]"))
            )
            driver.execute_script("arguments[0].click();", videos_tab)
            time.sleep(5)

            video_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, '//a[@id="video-title"]'))
            )

            if len(video_elements) < num_videos:
                return jsonify({"error": f"El canal solo tiene {len(video_elements)} videos"})

            selected_video = random.choice(video_elements)
            video_url = selected_video.get_attribute("href")
            driver.get(video_url)
            driver.maximize_window()
            time.sleep(2)

            title = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.title.style-scope.ytd-video-primary-info-renderer'))
            ).text

            SCROLL_PAUSE_TIME = 2
            CYCLES = 20
            html = driver.find_element(By.TAG_NAME, 'html')

            with tqdm(total=CYCLES, desc="Cargando comentarios") as pbar:
                for _ in range(CYCLES):
                    html.send_keys(Keys.END)
                    time.sleep(SCROLL_PAUSE_TIME)
                    pbar.update(1)

            time.sleep(5)
            try:
                comment_elems = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, '#content-text'))
                )
                video_comments = [elem.text for elem in comment_elems]
                for comment in video_comments:
                    comments.append({"title": title, "url": video_url, "comment": comment})
            except StaleElementReferenceException:
                driver.refresh()
                time.sleep(2)
                comment_elems = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, '#content-text'))
                )
                video_comments = [elem.text for elem in comment_elems]
                for comment in video_comments:
                    comments.append({"title": title, "url": video_url, "comment": comment})

    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if 'driver' in locals():
            driver.quit()

    output_file = 'yt_comments.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url", "comment"])
        writer.writeheader()
        writer.writerows(comments)

    # Preprocesar los datos
    dfOriginal = pd.read_csv(output_file, sep=',', low_memory=False)
    dataframe = copy.deepcopy(dfOriginal)
    dataframe = dataframe.drop(['title'], axis=1)
    dataframe = dataframe.drop(['url'], axis=1)
    dataframe["comment"] = dataframe["comment"].str.lower()

    # Funciones de preprocesamiento
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize_text(text):
        if not isinstance(text, str):
            return []
        return word_tokenize(text)

    def clean_text(tokens):
        text = " ".join(tokens)
        text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s,]', '', text, flags=re.UNICODE)
        text = re.sub(r'\d+', '', text)
        text = remove_punctuation(text)
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords.words('spanish')]
        return ' '.join(filtered_words)

    def lemmatize_text(text):
        doc = nlp(text)
        lemmatized_words = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_words)

    dataframe['comment_tokenized'] = dataframe['comment'].apply(tokenize_text)
    dataframe['comment_cleaned'] = dataframe['comment_tokenized'].apply(clean_text)
    dataframe['comment_lemmatized'] = dataframe['comment_cleaned'].apply(lemmatize_text)
    dataframe = dataframe.drop(['comment'], axis=1)
    dataframe = dataframe.drop(['comment_tokenized'], axis=1)
    dataframe = dataframe.drop(['comment_cleaned'], axis=1)

    preprocessed_output_file = 'preprocessed_yt_comments.csv'
    dataframe.to_csv(preprocessed_output_file, index=False)

    end_time = time.time()  # Finalizar el temporizador
    elapsed_time = end_time - start_time  # Calcular el tiempo transcurrido

    return jsonify({"message": f"Se han extraído y preprocesado comentarios de {num_videos} videos. Datos guardados en '{preprocessed_output_file}'. Tiempo tomado: {elapsed_time:.2f} segundos."})


@app.route('/process_csv', methods=['POST'])
def process_csv():
    # Leer el archivo CSV con los comentarios
    df = pd.read_csv('preprocessed_yt_comments.csv')  # Asegúrate de que la columna con los comentarios se llama 'comment_lemmatized'

    # Manejar valores nulos: eliminar filas con valores nulos en la columna 'comment_lemmatized'
    df = df.dropna(subset=['comment_lemmatized'])
    comentarios = df['comment_lemmatized']

    # Paso 2: Preprocesar los comentarios
    # Cargar el vectorizador TF-IDF usado durante el entrenamiento
    tfidf_vectorizer = joblib.load('preprocesamiento/vectorizer.pkl')  # Asegúrate de haber guardado el vectorizador durante el entrenamiento
    X_tfidf = tfidf_vectorizer.transform(comentarios)

    # Paso 3: Cargar el modelo entrenado
    best_rf_classifier = joblib.load('preprocesamiento/mejor_modelo_rf.pkl')

    # Paso 4: Hacer predicciones con el modelo
    predicciones = best_rf_classifier.predict(X_tfidf)

    # Asegúrate de que predicciones es un array unidimensional
    if predicciones.ndim > 1:
        predicciones = predicciones.argmax(axis=1)

    # Define el mapeo de índices a nombres de clases
    class_mapping = {
        0: 'scared',
        1: 'mad',
        2: 'sad',
        3: 'peaceful',
        4: 'powerful',
        5: 'joyful'
    }

    # Convertir las predicciones numéricas a nombres de clases
    predicciones_con_nombres = [class_mapping[pred] for pred in predicciones]

    # Paso 5: Agregar las predicciones numéricas y con nombres al DataFrame original
    df['predicciones_numericas'] = predicciones
    df['predicciones_nombres'] = predicciones_con_nombres

    # Guardar los resultados en un nuevo archivo CSV
    df.to_csv('static/comentarios_con_predicciones.csv', index=False)
    print("Predicciones guardadas exitosamente en 'static/comentarios_con_predicciones.csv'")

    # Generar la gráfica y guardarla como imagen
    plt.figure(figsize=(10, 6))
    sns.countplot(x='predicciones_nombres', data=df, order=list(class_mapping.values()))
    plt.title('Distribución de las Clases Predichas')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Frecuencia')
    plt_path = os.path.join('static', 'predicciones_grafica.png')
    plt.savefig(plt_path)
    plt.close()

    # Crear la tabla interactiva usando Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    table_path = os.path.join('static', 'predicciones_tabla.html')
    fig.write_html(table_path)

    return render_template('index.html', graph_image='predicciones_grafica.png', table_html='predicciones_tabla.html')


if __name__ == '__main__':
    app.run(debug=True)
