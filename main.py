import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analiza locuinte san francisco", layout="wide")
st.title("Analiza pietei imobiliare din san francisco")


@st.cache_data
def load_data():
    return pd.read_csv("date_sf.csv")


df = load_data()

st.sidebar.header("meniu proiect")
sectiune = st.sidebar.radio("alege un capitol:", [
    "1. explorare si curatare",
    "2. agregari statistice",
    "3. transformari date",
    "4. harta spatiala",
    "5. machine learning"
])

if sectiune == "1. explorare si curatare":
    st.header("1. tratarea valorilor lipsa si extreme")
    st.write("o mica parte din datele originale:")
    st.dataframe(df.head())

    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
    st.write("notificare: valorile lipsa pentru bai si dormitoare au fost completate cu mediana.")

    q1 = df['lastsoldprice'].quantile(0.25)
    q3 = df['lastsoldprice'].quantile(0.75)
    iqr = q3 - q1
    limita = q3 + 1.5 * iqr
    df_curat = df[df['lastsoldprice'] <= limita]

    st.write("notificare: au fost eliminate proprietatile cu preturi aberante, mult peste medie.")
    st.write("dimensiune set de date initial:", len(df))
    st.write("dimensiune dupa eliminarea extremelor:", len(df_curat))

elif sectiune == "2. agregari statistice":
    st.header("2. agregari folosind functii de grup (pandas)")
    st.write("grupam datele dupa cartier si calculam pretul mediu si suprafata maxima.")

    grup = df.groupby('neighborhood').agg({
        'lastsoldprice': 'mean',
        'finishedsqft': 'max',
        'zpid': 'count'
    }).rename(columns={'lastsoldprice': 'pret_mediu', 'finishedsqft': 'suprafata_maxima', 'zpid': 'numar_case'})

    st.dataframe(grup)
    st.write("grafic cu pretul mediu pe cartier:")
    st.bar_chart(grup['pret_mediu'])

elif sectiune == "3. transformari date":
    st.header("3. codificare si scalare")

    st.write("codificam tipul de cladire (usecode) in format numeric:")
    df_codificat = pd.get_dummies(df, columns=['usecode'], drop_first=True)
    st.dataframe(df_codificat.head())

    st.write("scalam suprafata, numarul de camere si baile pentru ca distributia sa fie uniforma:")
    scaler = StandardScaler()
    coloane = ['finishedsqft', 'totalrooms', 'bathrooms']
    df_scalat = df.copy()
    df_scalat[coloane] = scaler.fit_transform(df[coloane].fillna(0))
    st.dataframe(df_scalat[coloane].head())

elif sectiune == "4. harta spatiala":
    st.header("4. analiza cu geopandas")
    st.write("transformam latitudinea si longitudinea intr-o geometrie spatiala.")

    df_geo = df.dropna(subset=['latitude', 'longitude']).head(1000)
    geometrie = [Point(xy) for xy in zip(df_geo['longitude'], df_geo['latitude'])]
    gdf = gpd.GeoDataFrame(df_geo, geometry=geometrie)

    st.dataframe(gdf[['neighborhood', 'lastsoldprice', 'geometry']].head())
    st.write("distributia caselor pe harta:")
    st.map(df_geo[['latitude', 'longitude']])

elif sectiune == "5. machine learning":
    st.header("5. clusterizare si regresie")

    date_ml = df.dropna(subset=['latitude', 'longitude', 'lastsoldprice', 'finishedsqft', 'bathrooms', 'totalrooms'])

    st.subheader("a. clusterizare k-means")
    st.write("impartim casele in 3 zone investitionale pe baza de locatie si pret:")

    model_kmeans = KMeans(n_clusters=3, random_state=42)
    date_ml['cluster'] = model_kmeans.fit_predict(date_ml[['latitude', 'longitude', 'lastsoldprice']])

    fig, ax = plt.subplots()
    ax.scatter(date_ml['longitude'], date_ml['latitude'], c=date_ml['cluster'], cmap='viridis', s=15)
    st.pyplot(fig)

    st.subheader("b. regresie multipla")
    st.write("estimam pretul casei avand ca variabile suprafata, numarul de bai si numarul de camere.")

    x = date_ml[['finishedsqft', 'bathrooms', 'totalrooms']]
    x = sm.add_constant(x)
    y = date_ml['lastsoldprice']

    model_regresie = sm.OLS(y, x).fit()
    st.text(model_regresie.summary())