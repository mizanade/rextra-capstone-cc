import pandas as pd
import numpy as np
import re
import os
import nltk
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

# Download stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess CV text
def preprocess_cv(text1):
    # Lowercase
    text1 = text1.lower()
    # Remove symbols
    pattern = r'[^A-Za-z\s]'
    text1 = re.sub(pattern, '', text1)
    # Remove stopwords
    stop = set(stopwords.words('english'))
    text1 = ' '.join([word for word in text1.split() if word not in stop])
    return text1

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ''
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading PDF file '{file_path}': {str(e)}")
    return text

# Load the dataset
file_path = 'dataset-new.csv'
df = pd.read_csv(file_path)

# Function to remove emoticons
def remove_emoticons(text):
    emoticon_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251"
                                  "]+", flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text).replace('\n\n', ' ')

# Apply emoticon removal
df_clean = df.copy()
df_clean['Deskripsi'] = df_clean['Deskripsi'].apply(remove_emoticons)
df_clean['Skill yang didapat'] = df_clean['Skill yang didapat'].apply(remove_emoticons)

# Remove stopwords
stop = set(stopwords.words('indonesian'))
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop])

df_clean['Deskripsi'] = df_clean['Deskripsi'].apply(remove_stopwords)
df_clean['Skill yang didapat'] = df_clean['Skill yang didapat'].apply(remove_stopwords)

# Remove symbols and convert to lowercase
pattern = r'[$-/:-?{-~!"^_`\[\]]'
df_clean['Deskripsi'] = df_clean['Deskripsi'].str.replace(pattern, '', regex=True).str.lower()
df_clean['Skill yang didapat'] = df_clean['Skill yang didapat'].str.replace(pattern, '', regex=True).str.lower()

# Combine relevant columns into one
df_clean['combined'] = df_clean['Deskripsi'] + ' ' + df_clean['Skill yang didapat']

# Define lemmatization function
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df_clean['combined'] = df_clean['combined'].apply(lemmatize)

# Initialize vectorizer and calculate TF-IDF vectors
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
X = vectorizer.fit_transform(df_clean['combined']).toarray()

# Dummy target variable (replace with actual labels if available)
y = np.random.randint(0, 2, X.shape[0])

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained model
model_path = 'model.h5'
model = load_model(model_path)

from tensorflow.keras.regularizers import l2

# TensorFlow-based neural network model
# def create_model(input_dim):
#     model = Sequential([
#         Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)),
#         Dropout(0.5),
#         Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
#         Dropout(0.5),
#         Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#         Dropout(0.5),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# Create and train the model
# model = create_model(X_train.shape[1])
# model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))


# Function to clean user input
def clean_input(user_input):
    user_input = re.sub(r'[^\w\s]', '', user_input)
    user_input = user_input.lower()
    return user_input

# Recommendation function
def recommend_activities(user_input, user_preference, user_kategori_khusus, penyelenggara, durasi, skills, past_activities, preprocessed_cv, top_n=3):
    # Filter based on general preference
    if user_preference.lower() == 'dalam kampus':
        filtered_df = df_clean[df_clean['Kategori Umum'] == 'Kegiatan Intra Kampus']
        # Further filter by penyelenggara if specified
        if penyelenggara:
            filtered_df = filtered_df[filtered_df['Penyelenggara'] == penyelenggara]
    else:
        filtered_df = df_clean[df_clean['Kategori Umum'] == 'Kegiatan Umum']

    # Further filter by kategori khusus if specified
    if user_kategori_khusus:
        filtered_df = filtered_df[filtered_df['Kategori Khusus'] == user_kategori_khusus]

    # Further filter by durasi if specified
    if durasi:
        filtered_df = filtered_df[filtered_df['Durasi'] == durasi]

    if filtered_df.empty:
        return []

    filtered_combined = filtered_df['combined']
    filtered_vectors = vectorizer.transform(filtered_combined)

    user_input_vector = vectorizer.transform([user_input])
    skills_vector = vectorizer.transform([skills])
    past_activities_vector = vectorizer.transform([past_activities])
    cv_vector = vectorizer.transform([preprocessed_cv])

    user_combined_vector = user_input_vector + skills_vector + past_activities_vector + cv_vector

    similarity_scores = cosine_similarity(user_combined_vector, filtered_vectors).flatten()

    # top_indices = [index for index in similarity_scores.argsort()[::-1]][:top_n]

    # Get the top N recommendations with similarity score above threshold
    top_indices = [index for index in similarity_scores.argsort()[::-1] if similarity_scores[index] > 0.1][:top_n]

    recommendations = []
    for index in top_indices:
        recommendations.append({
            'id': filtered_df.iloc[index]['ID'],
            'posisi': filtered_df.iloc[index]['Posisi'] if 'Posisi' in filtered_df.columns else 'N/A',
            'penyelenggara': filtered_df.iloc[index]['Penyelenggara'] if 'Penyelenggara' in filtered_df.columns else 'N/A',
            'deskripsi': filtered_df.iloc[index]['Deskripsi'],
            'durasi': filtered_df.iloc[index]['Durasi'],
            'persyaratan': filtered_df.iloc[index]['Persyaratan Pendaftaran'] if 'Persyaratan Pendaftaran' in filtered_df.columns else 'N/A',
            'skills': filtered_df.iloc[index]['Skill yang didapat'],
            'lokasi': filtered_df.iloc[index]['Lokasi'] if 'Lokasi' in filtered_df.columns else 'N/A',
            'similarity_score': similarity_scores[index]
        })
    return recommendations


# Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('form.html')

@app.route('/recommend', methods=['POST'])
def recommend():
        
    user_preference = request.form['preference']
    user_kategori_khusus = request.form['user_preference']
    user_skills = request.form['skills']
    user_past_activities = request.form['past_activities']
    penyelenggara = request.form['penyelenggara']
    durasi = request.form['durasi']

    user_input = clean_input(f"{user_preference} {user_kategori_khusus} {user_skills} {user_past_activities} {penyelenggara} {durasi}")
    print(user_input)

    cv_file = request.files.get('cv')
    preprocessed_cv = ""
    if cv_file and cv_file.filename != '':
        file_path = os.path.join('uploads', cv_file.filename)
        cv_file.save(file_path)
        cv_text = extract_text_from_pdf(file_path)
        os.remove(file_path)  # Remove the temporary file
        preprocessed_cv = preprocess_cv(cv_text)
    print('tes')
    print(preprocessed_cv)

    recommendations = recommend_activities(user_input, user_preference, user_kategori_khusus, penyelenggara, durasi, user_skills, user_past_activities, preprocessed_cv)
    print(recommendations)
    for rec in recommendations:
        rec['id'] = int(rec['id'])
        rec['durasi'] = str(rec['durasi'])
    # print(recommendations)
    return jsonify({
        "status": "success",
        "message": "response succcess",
        "data": recommendations,
        }), 200


# Run the app
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8080)
