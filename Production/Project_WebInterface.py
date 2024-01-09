import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import folium
from streamlit_folium import folium_static
from PIL import Image

@st.cache_data 
def load_data():
    path = "C:/Users/emine/Documents/Ecole/A5/Webscraping & Data Processing/Project/data/Startup database.xlsx"
    data = pd.read_excel(path)
    return data

df = load_data()


with st.sidebar:
    image = Image.open('C:\\Users\\emine\\Documents\\Ecole\\A5\\Webscraping & Data Processing\\Project\\Logo.png')
    st.image(image)

    selected = option_menu("Main Menu", ["Startups Informations", "Startups Locations", "Startups Scores"], 
                           icons=['house', 'map', 'trophy'], menu_icon="cast", default_index=0)

if selected == "Startups Informations":
    st.title("Startups Informations")

    
    sector = st.selectbox('Choisissez un secteur', df['sector'].unique())

    
    filtered_data = df[df['sector'] == sector]

    
    st.write(filtered_data[['name', 'phone', 'size', 'website', 'founded', 'location']])

elif selected == "Startups Locations":
    st.title("Startups Locations")

    sector_map = st.selectbox('Choisissez un secteur pour la carte', df['sector'].unique(), key='sector_map')

    
    filtered_data_map = df[df['sector'] == sector_map]

    
    map = folium.Map(location=[20, 0], zoom_start=2)
    for _, row in filtered_data_map.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=row['name'],
                icon=folium.Icon(color='blue')
            ).add_to(map)

    
    folium_static(map)

elif selected == "Startups Scores":
    st.title("Startups Scores")

    
    all_words = set()
    for specialization in df['specialisation'].dropna():
        words = specialization.split()
        all_words.update(words)

    
    unique_words = list(all_words)
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.spatial.distance import cosine
    from sentence_transformers import SentenceTransformer
    import numpy as np

    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    
    rse_keywords = ["sustainability", "ethical", "social", "environment", "health", "well-being", "community", 
                    "green", "renewable", "eco-friendly", "inclusive", "diversity", "equality", "charity", "volunteer"]


    
    rse_embeddings = model.encode(rse_keywords)
    input_embeddings = model.encode(unique_words)

    
    threshold = 0.65
    rse_related_words = []

    for word, word_embedding in zip(unique_words, input_embeddings):
        similarities = [1 - cosine(word_embedding, rse_embedding) for rse_embedding in rse_embeddings]
        if max(similarities) > threshold:
            rse_related_words.append(word)

    
    def calculate_rse_score(specialisation):
        if pd.isna(specialisation):
            return 0
        words = specialisation.split()
        return sum(word in rse_related_words for word in words)

    df['rse_score'] = df['specialisation'].apply(calculate_rse_score)

    
    sector_score = st.selectbox('Choisissez un secteur pour le score RSE', df['sector'].unique(), key='sector_score')

    
    filtered_data_score = df[df['sector'] == sector_score]

    
    st.write(filtered_data_score[['name', 'rse_score']])
