import pandas as pd

from location_ml import cluster_places, recommend_places


def sample_places() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "Near Cafe", "category": "Cafe", "latitude": 12.9717, "longitude": 77.5947, "rating": 4.1},
            {"name": "Far Cafe", "category": "Cafe", "latitude": 13.10, "longitude": 77.70, "rating": 4.9},
            {"name": "Near Park", "category": "Park", "latitude": 12.9718, "longitude": 77.5948, "rating": 4.5},
        ]
    )


def test_recommendations_are_nearest_and_respect_filters() -> None:
    results = recommend_places(sample_places(), 12.9716, 77.5946, ["Cafe"], radius_km=2)

    assert results["name"].tolist() == ["Near Cafe"]
    assert results.iloc[0]["distance_km"] < 1


def test_recommendations_return_empty_for_no_category_match() -> None:
    results = recommend_places(sample_places(), 12.9716, 77.5946, ["Hospital"], radius_km=2)

    assert results.empty
    assert "distance_km" in results.columns


def test_empty_category_selection_returns_no_recommendations_or_clusters() -> None:
    places = sample_places()
    no_categories = places[places["category"].isin([])]

    recommendations = recommend_places(places, 12.9716, 77.5946, [], radius_km=2)
    clusters = cluster_places(no_categories)

    assert recommendations.empty
    assert clusters.empty
    assert "cluster" in clusters.columns


def test_dbscan_groups_nearby_places_and_marks_distant_place_as_noise() -> None:
    results = cluster_places(sample_places(), radius_km=1, min_samples=2)

    assert results.loc[0, "cluster"] == results.loc[2, "cluster"]
    assert results.loc[1, "cluster"] == -1