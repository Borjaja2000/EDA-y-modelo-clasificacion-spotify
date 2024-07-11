import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pydeck as pdk
import seaborn as sns
import plotly.graph_objects as go

import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
import streamlit.components.v1 as components

logo = 'spotify.png'
st.set_page_config(page_title="Spotify", page_icon=logo ,layout="wide") #configuración de la página


#Cargar datos
df = pd.read_csv("spotify_data_cleaned.zip", low_memory=False)

#Funciones
#Función para limpiar los outliers de las columnas que indiquemos
def clean_outliers(df_aux, columns: list)->pd.DataFrame:
    for column in columns:
        Q1 = df_aux[column].quantile(0.25)
        Q3 = df_aux[column].quantile(0.75)
        IQR = Q3 - Q1
        df_aux = df_aux[(df_aux[column] >= Q1-1.5*IQR) & (df_aux[column] <= Q3 + 1.5*IQR)]
    return df_aux


st.title("Análisis exploratorio de canciones de Spotify")
st.sidebar.title("Opciones de la tabla")
pestaña = st.sidebar.radio("Selecciona una pestaña:", ("Inicio", "Distribución variables", "Popularidad", "Características de la canción", "Importancia del rating"))
numero_artistas = st.sidebar.slider("Número de artistas", 1, 50, 10, key="artistas")
numero_canciones = st.sidebar.slider("Número de canciones", 1, 50, 10, key="canciones")

if pestaña == "Inicio":
    cols = st.columns(2)
    with cols[0]:
        pass
    with cols[1]:
        pass

elif pestaña == "Distribución variables":
    tabsInicio = st.tabs(["Variables continuas", "Correlación de Spearman"])
    with tabsInicio[0]:
        features = [
        ('popularity', 'Popularity'),
        ('energy', 'Energy'),
        ('danceability', 'Danceability'),
        ('loudness', 'Loudness'),
        ('speechiness', 'Speechiness'),
        ('acousticness', 'Acousticness'),
        ('instrumentalness', 'Instrumentalness'),
        ('liveness', 'Liveness'),
        ('valence', 'Valence'),
        ('tempo', 'Tempo')
        ]

        # Crear la figura y los ejes para los subplots
        fig, ax = plt.subplots(5, 2, figsize=(10, 6))

        # Iterar sobre las características y los ejes para llenar los subplots
        for i, (feature, title) in enumerate(features):
            row, col = divmod(i, 2)
            ax[row, col].hist(df[feature], bins=20, color="skyblue", edgecolor='black', linewidth=0.8)
            ax[row, col].title.set_text(title)

        # Ajustar el layout y mostrar el gráfico en Streamlit
        fig.tight_layout()
        st.pyplot(fig)

    with tabsInicio[1]:
        corr_matrix = df.corr(method='spearman', numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Mapa de calor de la Correlación de Spearman')
        st.pyplot(plt)

elif pestaña == "Popularidad":
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
                        hover_data={
                            'genre': True,  # Incluye 'genre' en el hover sin cambiar el nombre
                            'average_popularity': ':.2f'  # Formatea 'average_popularity'
                        })
        fig.update_traces(hovertemplate='Artista: %{label}<br>Popularidad Media: %{value:.2f}<br>Género(s): %{customdata[0]}')
        st.plotly_chart(fig)

        cols = st.columns(2)
        with cols[0]:
            fig = px.histogram(df, x='mode', y="popularity", title='Media de la popularidad en base al modo'
            , labels={"popularity": "Popularidad", "mode": "Modo"}
            , histfunc="avg")
            st.plotly_chart(fig)
        with cols[1]:
            fig = px.histogram(df, x='key', y="popularity", title='Media de la popularidad en base a la escala de la canción'
            , labels={"popularity": "Popularidad", "key": "Escala"}
            , color="key"
            , histfunc="avg"
            , category_orders={"key": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]})
            st.plotly_chart(fig)

        df_aux = df.sort_values(by='popularity', ascending=False).head(numero_canciones)
        fig = px.parallel_categories(df_aux, dimensions=['genre', 'key', 'mode', 'popularity'], color="popularity", color_continuous_scale=px.colors.sequential.Agsunset
                                    , title=f'Top {numero_canciones} canciones y su camino hacia la popularidad')
        st.plotly_chart(fig)

    with tabsPrecio[1]:
        # Graficar las canciones más populares y su danceability
        df_aux = df.sort_values(by='popularity', ascending=False).head(numero_canciones)
        fig = px.area(df_aux, x='track_name', y='danceability', title=f'Top {numero_canciones} canciones con mayor popularidad y su danceability'
                , hover_data=["artist_name", "popularity"], labels={"danceability": "Danceability", "track_name": "Canción", "artist_name": "Artista", "popularity": "Popularidad"}
                , markers=True)
        st.plotly_chart(fig)
    with tabsPrecio[2]:
        #Histograma de popularidad media por género y año
        fig = px.histogram(y=df['popularity'], x=df['genre'], histfunc="avg"
                    , animation_frame=df["year"], title="Popularity by Genre and Year"
                    , labels={"y": "Popularidad", "x": "Género", "animation_frame": "Año"}
                    , category_orders={"animation_frame": [2000, 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]}
                    )
        fig.update_layout(xaxis_tickfont_size=11)
        fig.update_xaxes(categoryorder="total ascending", tickangle=-35, title_standoff=0)
        st.plotly_chart(fig)
    with tabsPrecio[3]:
        fig = px.histogram(df, x='popularity', y="energy", title='Media de la energía en base a la popularidad según el año'
        ,labels={"popularity": "Popularidad", "energy": "Energía", "year": "Año"}
        ,animation_frame="year"
        ,category_orders={"year": [2000, 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]}
        ,range_y=[0, 1]
        ,range_x=[0, 100]
        ,histfunc="avg")
        st.plotly_chart(fig)
    with tabsPrecio[4]:
        fig = px.histogram(df, x='popularity', y="valence", title='Media de la positividad en base a la popularidad según el año'
        ,labels={"popularity": "Popularidad", "valence": "Positividad", "year": "Año"}
        ,animation_frame="year"
        ,category_orders={"year": [2000, 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]}
        ,range_y=[0, 1]
        ,range_x=[0, 100]
        ,histfunc="avg")
        st.plotly_chart(fig)
elif pestaña == "Características de la canción":
    tabsVecindario = st.tabs(["Artistas", "Volumen", "Tempo", "Según puntuación de ubicación"])
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
                        title=f'Top {numero_artistas} artistas con más canciones')
        fig.update_traces(hovertemplate='Artista: %{label}<br>Número de Canciones: %{value}')
        st.plotly_chart(fig)
    with tabsVecindario[1]:
        df_aux = df
        df_aux['loudness_color'] = df_aux['loudness'].apply(lambda x: 'Menor que cero' if x < 0 else 'Mayor o igual a cero')
        color_map = {'Menor que cero': '#add8e6', 'Mayor o igual a cero': '#2874A6 '}

        fig = px.histogram(df_aux, x='loudness', y="energy", histfunc='avg', 
                        title='Histograma del Volumen con la Energía Promedio',
                        color='loudness_color',
                        labels={'loudness': 'Volumen', 'energy': 'Energía'},
                        color_discrete_map=color_map,
                        nbins=30) 

        box = go.Figure(go.Box(x=df['loudness'], boxmean=True, name="Boxplot", marker_color="#cccccc"))
        for trace in box.data:
            fig.add_trace(go.Box(x=trace['x'], boxmean=True, marker_color="#148F77"))
        fig.update_layout(xaxis_title='Loudness', yaxis_title='Energy', showlegend=False)
        st.plotly_chart(fig)

    with tabsVecindario[2]:
        df_aux = clean_outliers(df, ['tempo'])
        fig = px.histogram(df_aux, x='tempo', y='danceability', title=f'Bailabilidad en base al tempo de las canciones'
                , hover_data=["artist_name", "popularity"]
                , labels={"danceability": "Danceability", "track_name": "Canción", "artist_name": "Artista", "popularity": "Popularidad"}
                , histfunc="avg")
        st.plotly_chart(fig)
    with tabsVecindario[3]:
        pass    
elif pestaña == "Importancia del rating":
    codigo_iframe = ''''''
    components.html(codigo_iframe, width=1320, height=1250)















