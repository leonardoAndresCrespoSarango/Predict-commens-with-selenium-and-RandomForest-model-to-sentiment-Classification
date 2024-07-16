import pandas as pd
import copy
import string
import re
from multiprocessing import Pool, cpu_count
import time
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
# Cargar el modelo de lenguaje en español de spaCy
nlp = spacy.load('es_core_news_sm')

stopwords_es = stopwords.words('spanish')
lemmatizer = WordNetLemmatizer()
PUNCT_TO_REMOVE = string.punctuation
dfOriginal = pd.read_csv('C:/Users/lcres/PycharmProjects/SELENIUM/yt_comments.csv', sep = ',',low_memory=False)

dataframe=copy.deepcopy(dfOriginal)
print(dataframe.shape)
dataframe.head(5)

dataframe=dataframe.drop(['title'], axis=1)
dataframe=dataframe.drop(['url'], axis=1)
print(dataframe.shape)
print(dataframe.head(10))
dataframe["comment"] = dataframe["comment"].str.lower()
from nltk.tokenize import word_tokenize
# Función para remover puntuación
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
# Función para tokenizar el texto
def tokenize_text(text):
    return word_tokenize(text)

# Función para limpiar el texto

def clean_text(tokens):
    text = " ".join(tokens)
    # Eliminar URLs
    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
    # Eliminar emojis (opción mejorada que no elimina tildes)
    text = re.sub(r'[^\w\s,]', '', text, flags=re.UNICODE)
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    # Eliminar puntuación
    text = remove_punctuation(text)
    words = text.split()
    # Filtrar palabras vacías
    filtered_words = [word for word in words if word.lower() not in stopwords_es]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)
# Función para procesar un bloque del DataFrame
def process_block(block):
    print("Procesando..............")
    block['comment_tokenized'] = block['comment'].apply(tokenize_text)
    print("Texto tokenizado")
    block['comment_cleaned'] = block['comment_tokenized'].apply(clean_text)
    print("Texto limpio")
    block['comment_lemmatized'] = block['comment_cleaned'].apply(lemmatize_text)
    print("Texto lematizado")
    return block
# Dividir el DataFrame en bloques y procesar en paralelo
num_blocks = cpu_count()
blocks = [dataframe.iloc[i::num_blocks] for i in range(num_blocks)]
start_time_parallel = time.time()
with Pool(num_blocks) as pool:
    result_blocks = pool.map(process_block, blocks)
end_time_parallel = time.time()
result_df = pd.concat(result_blocks)
# Guarda el DataFrame en un archivo CSV
result_df.to_csv('comentarios-preprocesados.csv', index=False)
