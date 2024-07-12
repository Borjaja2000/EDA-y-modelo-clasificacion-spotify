import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import streamlit.components.v1 as components

logo = 'spotify.png'
st.set_page_config(page_title="Spotify", page_icon=logo ,layout="wide") #configuración de la página
#Funciones
@st.cache_data(ttl=3600)
def cargar_datos():
    df = pd.read_csv("spotify_data_cleaned.zip", low_memory=False)
    return df

def clean_outliers(df_aux, columns: list)->pd.DataFrame:
    for column in columns:
        Q1 = df_aux[column].quantile(0.25)
        Q3 = df_aux[column].quantile(0.75)
        IQR = Q3 - Q1
        df_aux = df_aux[(df_aux[column] >= Q1-1.5*IQR) & (df_aux[column] <= Q3 + 1.5*IQR)]
    return df_aux

df = cargar_datos()

st.title("Análisis exploratorio de canciones de Spotify")
st.sidebar.title("Opciones de la tabla")
pestaña = st.sidebar.radio("Selecciona una pestaña:", ("Inicio", "Distribución variables", "Popularidad", "Características de la canción", "Importancia del rating"))


if pestaña == "Inicio":
    cols = st.columns(2)
    with cols[0]:
        pass
    with cols[1]:
        pass

elif pestaña == "Distribución variables":
    tabsInicio = st.tabs(["Variables continuas", "Correlación de Spearman"])
    with tabsInicio[0]:
        st.image('imagenes/distribucion.png')
    with tabsInicio[1]:
        st.image('imagenes/spearman.png')

elif pestaña == "Popularidad":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configuración")
    numero_artistas = st.sidebar.slider("Número de artistas", 1, 50, 10, key="artistas")
    numero_canciones = st.sidebar.slider("Número de canciones", 1, 50, 10, key="canciones")

    tabsPrecio = st.tabs([f"Top Artistas y Canciones", "Bailable", "Género", "Energía", "Positividad"])
    with tabsPrecio[0]:
        df['genre'] = df['genre'].apply(lambda x: x.capitalize())
        artist_info = df.groupby('artist_name').agg({
            'popularity': 'mean',
            'genre': 'first'  # Concatena géneros únicos
        }).reset_index()
        artist_info.rename(columns={'popularity': 'average_popularity'}, inplace=True)
        top_n_artists = artist_info.sort_values(by='average_popularity', ascending=False).head(numero_artistas)

        fig = px.treemap(top_n_artists, 
                        path=['artist_name'], 
                        values='average_popularity',
                        color='average_popularity', 
                        color_continuous_scale='RdYlGn',
                        title=f'Top {numero_artistas} Artistas con la media más alta de popularidad',
                        custom_data=['genre'],
                        labels={'average_popularity': 'Popularidad Media', 'artist_name': 'Artista', 'genre': 'Género'})
        fig.update_traces(hovertemplate='Artista: %{label}<br>Popularidad Media: %{value:.2f}<br>Género(s): %{customdata[0]}')
        st.plotly_chart(fig)

        cols = st.columns(2)
        with cols[0]:
            with open("html/modopopularidad.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)
        with cols[1]:
            with open("html/escalapopularidad.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)

        df_aux = df.sort_values(by='popularity', ascending=False).head(numero_canciones)
        fig = px.parallel_categories(df_aux
                                    ,dimensions=['genre', 'key', 'mode', 'popularity']
                                    ,color="popularity"
                                    ,color_continuous_scale=px.colors.sequential.Agsunset
                                    ,title=f'Top {numero_canciones} canciones y su camino hacia la popularidad'
                                    ,labels={"genre": "Género", "key": "Key", "mode": "Modo", "popularity": "Popularidad"})
        st.plotly_chart(fig)

    with tabsPrecio[1]:
        # Graficar las canciones más populares y su danceability
        df_aux = df.sort_values(by='popularity', ascending=False).head(numero_canciones)
        fig = px.area(df_aux, x='track_name', y='danceability', title=f'Top {numero_canciones} canciones con mayor popularidad y su bailabilidad'
                , hover_data=["artist_name", "popularity"], labels={"danceability": "Bailabilidad", "track_name": "Canción", "artist_name": "Artista", "popularity": "Popularidad"}
                , markers=True)
        st.plotly_chart(fig)
    with tabsPrecio[2]:
        #Histograma de popularidad media por género y año
        with open("html/popularidadgeneros.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700)
    with tabsPrecio[3]:
        with open("html/energiapopularidad.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700)
    with tabsPrecio[4]:
        with open("html/positividadpopularidad.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700)
elif pestaña == "Características de la canción":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configuración")
    numero_artistas = st.sidebar.slider("Número de artistas", 1, 50, 10, key="artistas")
    tabsVecindario = st.tabs(["Artistas", "Volumen", "Tempo"])
    with tabsVecindario[0]:
        # Grafico artistas con mas canciones
        df_aux = df
        songs_per_artist = df_aux.groupby('artist_name', as_index=False)['track_name'].count()
        songs_per_artist.rename(columns={'track_name': 'song_count'}, inplace=True)
        top_50_artists = songs_per_artist.sort_values(by='song_count', ascending=False).head(numero_artistas)
        fig = px.treemap(top_50_artists, 
                        path=['artist_name'], 
                        values='song_count',
                        color='song_count', 
                        color_continuous_scale='RdYlGn',
                        title=f'Top {numero_artistas} artistas con más canciones',
                        labels={'song_count': 'Total Canciones'})
        fig.update_traces(hovertemplate='Artista: %{label}<br>Número de Canciones: %{value}')
        st.plotly_chart(fig)
    with tabsVecindario[1]:
        with open("html/volumenenergia.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700)

    with tabsVecindario[2]:
        with open("html/tempobailable.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500)
        st.image('imagenes/tempobailable.png')  
elif pestaña == "Importancia del rating":
    codigo_iframe = ''''''
    components.html(codigo_iframe, width=1320, height=1250)
