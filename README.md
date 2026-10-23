# GeoReach: Machine Learning for Location-Based Services

GeoReach is a final-year project prototype that demonstrates nearby-place recommendation and geographic hotspot discovery. It combines a Streamlit dashboard with scikit-learn's Haversine nearest-neighbor search and DBSCAN clustering.

## Features

- Recommend places near a latitude/longitude, with category and distance filters.
- Rank recommendations by geographic distance, using rating as a tie-breaker.
- Discover spatial clusters with DBSCAN and a configurable neighborhood radius.
- Inspect results on an interactive map and download recommendations as CSV.
- Run locally without API keys or external location services.

## Project structure

```text
.
|-- app.py
|-- location_ml.py
|-- data/
|   `-- places.csv
|-- tests/
|   `-- test_location_ml.py
|-- requirements.txt
`-- README.md
```

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

Run the tests with:

```powershell
python -m pytest
```

## Methodology

### Nearby recommendations

The reference location and place coordinates are converted from degrees to radians. A `NearestNeighbors` model with `metric="haversine"` finds geographic distances on a spherical Earth. Distances are converted to kilometers, filtered by the selected radius, and sorted nearest-first. Ratings break ties between equally distant places.

### Hotspot discovery

DBSCAN groups places whose pairwise Haversine distance is within the configured radius. The `min_samples` setting controls the minimum number of places needed to form a cluster. DBSCAN label `-1` indicates a point that is not part of a cluster.

## Data and limitations

`data/places.csv` contains synthetic demonstration records centered around Bengaluru. Names, coordinates, categories, and ratings are sample data, not verified business listings. For a research submission, replace it with a properly licensed dataset, document its source and collection date, and evaluate recommendations using a held-out set or user study. The current recommender is geographic, not personalized; ratings only break distance ties.

## Possible extensions

- Add a real, licensed POI dataset and a data-cleaning pipeline.
- Add user preferences and a content-based or collaborative ranking model.
- Compare Haversine nearest neighbors with a baseline and report precision@k or NDCG.
- Add temporal demand data and evaluate hotspot stability across time periods.
- Add privacy controls before collecting or storing users' location histories.