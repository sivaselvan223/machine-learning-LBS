"""GeoReach: location-based recommendations and hotspot analysis."""

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from location_ml import cluster_places, recommend_places

DATA_PATH = Path(__file__).parent / "data" / "places.csv"
CATEGORY_COLORS = {
    "Cafe": "blue",
    "Restaurant": "orange",
    "Hospital": "red",
    "Park": "green",
    "Shopping": "purple",
}
CLUSTER_COLORS = ["blue", "green", "orange", "purple", "darkred", "cadetblue", "darkgreen"]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_map(latitude: float, longitude: float, places: pd.DataFrame, show_radius_km: float | None = None) -> folium.Map:
    map_view = folium.Map(location=[latitude, longitude], zoom_start=13, control_scale=True)
    folium.Marker(
        [latitude, longitude],
        tooltip="Search location",
        icon=folium.Icon(color="black", icon="user", prefix="fa"),
    ).add_to(map_view)
    if show_radius_km is not None:
        folium.Circle(
            [latitude, longitude],
            radius=show_radius_km * 1000,
            color="#287a68",
            fill=True,
            fill_opacity=0.07,
        ).add_to(map_view)

    for _, place in places.iterrows():
        cluster = int(place["cluster"]) if "cluster" in place else None
        if cluster is not None:
            color = "gray" if cluster == -1 else CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
        else:
            color = CATEGORY_COLORS.get(place["category"], "blue")
        details = f"{place['name']} | {place['category']} | rating {place['rating']:.1f}"
        if "distance_km" in place:
            details += f" | {place['distance_km']:.2f} km"
        if cluster is not None:
            details += f" | cluster {cluster if cluster >= 0 else 'noise'}"
        folium.CircleMarker(
            [place["latitude"], place["longitude"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.85,
            tooltip=details,
        ).add_to(map_view)
    return map_view


st.set_page_config(page_title="GeoReach | Location Intelligence", layout="wide")
st.title("GeoReach")
st.caption("Location-based recommendations and spatial hotspot analysis")

places = load_data()
categories = sorted(places["category"].unique())

with st.sidebar:
    st.header("Search area")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.9716, format="%.5f")
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.5946, format="%.5f")
    mode = st.radio("Analysis", ["Recommendations", "Hotspot clusters"])
    st.divider()
    selected_categories = st.multiselect("Place categories", categories, default=categories)

st.info("Demo mode: the bundled points are synthetic and centered around Bengaluru. They are not real business listings.")

if mode == "Recommendations":
    radius_km = st.slider("Maximum distance", min_value=1, max_value=25, value=8, step=1)
    limit = st.slider("Places to show", min_value=5, max_value=30, value=15, step=5)
    results = recommend_places(places, latitude, longitude, selected_categories, radius_km, limit)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Matches", len(results))
    metric_columns[1].metric("Search radius", f"{radius_km} km")
    metric_columns[2].metric("Categories", len(selected_categories))

    map_column, results_column = st.columns([1.2, 1])
    with map_column:
        st.subheader("Nearby places")
        st_folium(build_map(latitude, longitude, results, radius_km), height=520, use_container_width=True)
    with results_column:
        st.subheader("Recommended results")
        if results.empty:
            st.warning("No places match these filters. Try a wider radius or more categories.")
        else:
            st.dataframe(
                results[["name", "category", "rating", "distance_km"]].rename(
                    columns={"name": "Place", "category": "Category", "rating": "Rating", "distance_km": "Distance (km)"}
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download recommendations as CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="georeach_recommendations.csv",
                mime="text/csv",
            )
else:
    radius_km = st.slider("Cluster neighborhood radius", min_value=1, max_value=5, value=2, step=1)
    min_samples = st.slider("Minimum places per cluster", min_value=2, max_value=8, value=3, step=1)
    selected = places[places["category"].isin(selected_categories)]
    clustered = cluster_places(selected, radius_km=radius_km, min_samples=min_samples)
    cluster_count = clustered.loc[clustered["cluster"] >= 0, "cluster"].nunique()

    metric_columns = st.columns(3)
    metric_columns[0].metric("Places analyzed", len(clustered))
    metric_columns[1].metric("Detected clusters", cluster_count)
    metric_columns[2].metric("Unclustered places", int((clustered["cluster"] == -1).sum()))

    map_column, summary_column = st.columns([1.2, 1])
    with map_column:
        st.subheader("Spatial clusters")
        st_folium(build_map(latitude, longitude, clustered), height=520, use_container_width=True)
    with summary_column:
        st.subheader("Cluster composition")
        composition = clustered[clustered["cluster"] >= 0].groupby("cluster").agg(
            Places=("name", "count"), Categories=("category", "nunique"), Average_rating=("rating", "mean")
        )
        if composition.empty:
            st.warning("No clusters found. Increase the radius or lower the minimum places setting.")
        else:
            st.dataframe(composition.round(2), use_container_width=True)
        st.caption("DBSCAN uses Haversine distance, so the neighborhood radius is measured in kilometers.")