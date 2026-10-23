"""Machine-learning utilities for location-based recommendations."""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

EARTH_RADIUS_KM = 6371.0088
REQUIRED_COLUMNS = {"name", "category", "latitude", "longitude", "rating"}


def validate_places(places: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(places.columns)
    if missing:
        raise ValueError(f"Places data is missing columns: {', '.join(sorted(missing))}")


def recommend_places(
    places: pd.DataFrame,
    latitude: float,
    longitude: float,
    categories: list[str] | None = None,
    radius_km: float = 10.0,
    limit: int = 10,
) -> pd.DataFrame:
    """Find nearby places with Haversine nearest-neighbor search."""
    validate_places(places)
    if radius_km < 0:
        raise ValueError("radius_km cannot be negative.")
    if limit <= 0:
        return places.iloc[0:0].assign(distance_km=pd.Series(dtype=float))

    candidates = places.copy()
    if categories is not None:
        candidates = candidates[candidates["category"].isin(categories)]
    if candidates.empty:
        return candidates.assign(distance_km=pd.Series(dtype=float))

    radians = candidates[["latitude", "longitude"]].to_numpy() * (3.141592653589793 / 180)
    reference = [[latitude * (3.141592653589793 / 180), longitude * (3.141592653589793 / 180)]]
    model = NearestNeighbors(n_neighbors=len(candidates), metric="haversine", algorithm="ball_tree")
    model.fit(radians)
    distances, indices = model.kneighbors(reference)

    results = candidates.iloc[indices[0]].copy()
    results["distance_km"] = distances[0] * EARTH_RADIUS_KM
    results = results[results["distance_km"] <= radius_km]
    return results.sort_values(["distance_km", "rating"], ascending=[True, False]).head(limit).reset_index(drop=True)


def cluster_places(
    places: pd.DataFrame,
    radius_km: float = 1.0,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Group nearby places with DBSCAN using a geographic distance metric."""
    validate_places(places)
    if radius_km <= 0:
        raise ValueError("radius_km must be greater than zero.")
    if min_samples <= 0:
        raise ValueError("min_samples must be greater than zero.")

    result = places.copy()
    if result.empty:
        result["cluster"] = pd.Series(dtype=int)
        return result

    radians = result[["latitude", "longitude"]].to_numpy() * (3.141592653589793 / 180)
    model = DBSCAN(
        eps=radius_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    result["cluster"] = model.fit_predict(radians)
    return result