import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import joblib

print("Módulos y clases importados")

# Cargar el dataset
df_cleaned = pd.read_csv("C:/Users/lcres/PycharmProjects/SELENIUM/dataframeFinal (1).csv")

# Actualizar el argumento sparse a sparse_output
encoder = OneHotEncoder(sparse_output=False)

# Ajustar y transformar los datos
salida = encoder.fit_transform(df_cleaned[['sentiment']])
salida_Y = pd.DataFrame(salida, columns=encoder.get_feature_names_out(['sentiment']))
print(salida_Y)

# Vectorizar los textos
vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9)
vectorizer.fit(df_cleaned['text_lemmatized'].values)

X_train, X_test, y_train, y_test = train_test_split(df_cleaned['text_lemmatized'].values, salida_Y, test_size=0.2)

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
joblib.dump(vectorizer, 'vectorizer.pkl')

# Configurar el modelo con los mejores hiperparámetros
best_params = {
    'bootstrap': False,
    'max_depth': None,
    'max_features': 'sqrt',  # Cambiado de 'auto' a 'sqrt'
    'min_samples_leaf': 1,
    'min_samples_split': 10,
    'n_estimators': 300
}

rf_classifier = RandomForestClassifier(**best_params)
rf_classifier.fit(X_train_tfidf, y_train)

# Guardar el modelo entrenado
joblib.dump(rf_classifier, 'mejor_modelo_rf.pkl')
print("Modelo guardado exitosamente en 'mejor_modelo_rf.pkl'")

# Evaluar el modelo
accuracy = rf_classifier.score(X_test_tfidf, y_test)
print(f"Accuracy del Random Forest optimizado: {accuracy:.2f}")

# Realizar predicciones en el conjunto de prueba
y_pred = rf_classifier.predict(X_test_tfidf)

# Convertir el formato multilabel-indicator a formato de una sola etiqueta
y_test_single = y_test.idxmax(axis=1)
y_pred_single = pd.DataFrame(y_pred, columns=y_test.columns).idxmax(axis=1)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_single, y_pred_single)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_single), yticklabels=np.unique(y_test_single))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Reporte de métricas (using the single-label format)
report = classification_report(y_test_single, y_pred_single)
print("Reporte de métricas:")
print(report)
