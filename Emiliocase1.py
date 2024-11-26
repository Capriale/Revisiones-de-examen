import pandas as pd

# Ejemplo de conjunto de datos
data = {
    'text': [
        "I am so happy today!",
        "I feel sad and lonely.",
        "This is frustrating.",
        "I am very angry right now.",
        "What a great day!",
        "I am feeling down.",
        "Everything is wonderful!",
        "I am mad at you."
    ],
    'emotion': [
        "Happy",
        "Sad",
        "Angry",
        "Angry",
        "Happy",
        "Sad",
        "Happy",
        "Angry"
    ]
}

# Crear un DataFrame
df = pd.DataFrame(data)

from sklearn.feature_extraction.text import CountVectorizer

# Crear un vectorizador
vectorizer = CountVectorizer()

# Convertir el texto a una matriz de características
X = vectorizer.fit_transform(df['text'])

# Obtener las etiquetas
y = df['emotion']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Imprimir resultados
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_emotion(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Ejemplo de predicción
new_message = "I feel amazing!"
predicted_emotion = predict_emotion(new_message)
print(f"The predicted emotion for the message: '{new_message}' is: {predicted_emotion}")
