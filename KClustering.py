# SECOND PARTIAL EXAM, ARTIFICIAL INTELLIGENCE
# Andres Cabral Reyes 172062

# Emotion Detection in Text using K-means Clustering
# This code uses Clustering (K means) to detect emotions by partitioning observations into clusters.
# The input will then be observed and then the algorithm will cluster it with the nearest mean (cluser centers or cluster centroid)
# In return, it will be able to determine which emotion the input shows.
# It's important to keep in mind that the algorithm is supervised. Therefore, the data must be already clasified

# STEPS TO SOLVE PROBLEM
# Begin by inserting pip install scikit-learn and pip install numpy into the terminal
# Make sure the libraries are installed in the correct environments. This can be verified with pip list 

# Import necessary libraries
# pip install pandas, scikit-learn, and numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Load the Kaggle dataset in CSV format
# Dataset used: https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset/data
dataset = pd.read_csv('/Users/andrescabral/Desktop/emotions.csv')

# Preview the dataset to understand its structure
print("\n")
print(dataset.head())

# The dataset has two columns: 'text' and 'emotions'
texts = dataset['text'].tolist()
labels = dataset['label'].tolist()

# Step 2: Vectorize the text data using TF-IDF
# This function converts text data into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
# It captures the importance of words relative to the dataset.
def vectorize_text(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts), vectorizer

# Vectorize both the text data and the input text
X, vectorizer = vectorize_text(texts)

# Step 3: Apply K-means clustering to group the texts into clusters
# This function applies K-means clustering on the vectorized data to form clusters.
# Each cluster should ideally represent one emotion.
def apply_kmeans(X, num_clusters):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

# Set the number of clusters to match the number of unique emotions
num_clusters = len(set(labels))
kmeans_model = apply_kmeans(X, num_clusters)

# Step 4: Map clusters to emotions based on the labeled data
# This function maps the clusters created by K-means to the corresponding emotions
# It checks with labeled sentences belong to certain clsuter and assigns the most frequent label to that cluster
def map_clusters_to_emotions(kmeans_model, labels):
    cluster_labels = kmeans_model.labels_
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    cluster_to_emotion = {}
    for i in range(num_clusters):
        # Find the most common label (emotion) in each cluster
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_emotion = np.bincount(encoded_labels[cluster_indices]).argmax()
        cluster_to_emotion[i] = label_encoder.inverse_transform([cluster_emotion])[0]
    
    return cluster_to_emotion

# Map clusters to emotions
cluster_emotion_mapping = map_clusters_to_emotions(kmeans_model, labels)

# Step 5: Detect the emotion of the user input text by vectorizing it, assigning it to a cluster, and returning the mapped emotion for that cluster
def detect_emotion(input_text, kmeans_model, vectorizer, cluster_emotion_mapping):
    input_vector = vectorizer.transform([input_text])  # Vectorize the input text
    cluster_label = kmeans_model.predict(input_vector)[0]  # Predict which cluster the text belongs to
    return cluster_emotion_mapping.get(cluster_label, "Unknown")

print("""\nEmotions:
0: Sadness
1: Joy
2: Love
3: Anger
4: Fear
5: Surprise
""")

# Input text
input_text = "i just feel really helpless and heavy hearted"
print(f"Input text: {input_text}")
# Run emotion detection on the input text
detected_emotion = detect_emotion(input_text, kmeans_model, vectorizer, cluster_emotion_mapping)
print(f"\nDetected emotion: {detected_emotion}")


#sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).

# Potential Errors:
# 1. Cluster Mismatch: K-means might assign similar words across different emotions to the same cluster.
#    This could lead to incorrect emotion detection, especially for sentences that contain mixed emotions.
# 
# 2. Input Text Variability: If the input text contains words not seen in the dataset, it may not cluster accurately.
#    Words that are entirely different or rarely used can result in a lack of meaningful context for the clustering algorithm.
#
# 3. Small Dataset Bias: If the training dataset is small or not representative of the diverse ways emotions are expressed,
#    the model may struggle to generalize and accurately classify input text. 
#
# 4. Overlapping Emotions: Some emotions may overlap significantly in language, making it difficult for K-means to separate them.
#    For instance, "happy" and "surprised" may be expressed with similar wording, leading to confusion.
#
# 5. High Dimensionality: TF-IDF vectorization may lead to high-dimensional data, which can affect K-means clustering performance.
#    The curse of dimensionality may make it harder for the algorithm to find distinct clusters.
#
# 6. Initialization Sensitivity: K-means is sensitive to the initial choice of centroids. Poor initialization can lead to suboptimal clustering.
#    Using methods like K-means++ for initialization can help mitigate this issue.
#
# 7. Assumption of Spherical Clusters: K-means assumes that clusters are spherical and evenly sized. If the actual clusters are not,
#    K-means may perform poorly in capturing the true structure of the data.

