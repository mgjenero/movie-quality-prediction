from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd


numeric_features: List[str] = [
    "budget",
    "gross",
    "duration",
    "num_critic_for_reviews",
    "num_user_for_reviews",
    "num_voted_users",
    "title_year",
]

categorical_features: List[str] = [
    "color",
    "genres",
    "language",
    "country",
    "content_rating",
]

y = ["is_good"]

selected_columns = numeric_features + categorical_features + y


def load_raw(path: str = "data/raw/movie_metadata.csv") -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def load_processed(path: str = "data/processed/movie_metadata_processed.csv") -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    # target
    df["is_good"] = (df["imdb_score"] >= 7).astype(int)
    # subset
    df_subset = df[selected_columns].copy()

    # Fill numeric NaNs with median
    for col in numeric_features:
        df_subset[col] = df_subset[col].fillna(df_subset[col].median())

    # Fill categorical NaNs with mode (if column exists)
    for col in categorical_features:
        df_subset[col] = df_subset[col].fillna(df_subset[col].mode()[0])

    # color -> binary
    df_subset["color"] = df_subset["color"].apply(
        lambda x: True if x == "Color" else False)

    # country fixes
    country_fix_map = {
        "West Germany": "Germany",
        "Soviet Union": "Russia",
        "Hong Kong": "China",
        "Official site": "Other",
        "New Line": "Other",

    }
    df_subset["country"] = df_subset["country"].replace(country_fix_map)

    return preprocess_normal(df_subset)


def preprocess_normal(df: pd.DataFrame) -> pd.DataFrame:

    allowed_countries = {'USA', 'UK', 'Other', 'France', 'Canada',
                         'Germany', 'Australia', 'China', 'India', 'Spain', 'Japan', 'Italy'}
    df["country"] = df["country"].where(
        df["country"].isin(allowed_countries), "Other")

    allowed_languages = {'English', 'Other',
                         'French', 'Spanish', 'Hindi', 'Mandarin'}
    df["language"] = df["language"].where(
        df["language"].isin(allowed_languages), "Other")

    rating_to_group = {
        "G": "Kids",
        "TV-Y": "Kids",
        "TV-Y7": "Kids",
        "TV-G": "Kids",
        "Approved": "Kids",
        "Passed": "Kids",
        "PG": "Young",
        "TV-PG": "Young",
        "GP": "Young",
        "M": "Young",
        "PG-13": "Teen",
        "TV-14": "Teen",
        "R": "Adult",
        "TV-MA": "Adult",
        "NC-17": "Explicit",
        "X": "Explicit",
        "Not Rated": "Other",
        "Unrated": "Other",
    }
    if "content_rating" in df.columns:
        df["content_rating"] = df["content_rating"].replace(
            rating_to_group)

    df["genres"] = df["genres"].apply(lambda x: x.split("|"))
    all_genres = sorted(
        {g for sublist in df["genres"] for g in sublist if g != ""})
    for genre in all_genres:
        df[f"genre_{genre}"] = df["genres"].apply(lambda lst: genre in lst)
    df = df.drop(columns=["genres"])

    categorical_simple = ["language", "country", "content_rating"]
    df = pd.get_dummies(df, columns=categorical_simple, dummy_na=False)

    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed/movies_processed2.csv") -> None:
    """Save the processed dataframe to CSV, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the data used for inference matches the training columns (one-hot encoding)."""
    df = preprocess_normal(df)  # Same preprocessing as training data
    train_columns = ['budget', 'gross', 'duration', 'num_critic_for_reviews', 'num_user_for_reviews', 'num_voted_users', 'title_year', 'color', 'genre_Action', 'genre_Adventure', 'genre_Animation', 'genre_Biography', 'genre_Comedy', 'genre_Crime', 'genre_Documentary', 'genre_Drama', 'genre_Family', 'genre_Fantasy', 'genre_Film-Noir', 'genre_Game-Show', 'genre_History', 'genre_Horror', 'genre_Music', 'genre_Musical', 'genre_Mystery', 'genre_News', 'genre_Reality-TV', 'genre_Romance', 'genre_Sci-Fi', 'genre_Short',
                     'genre_Sport', 'genre_Thriller', 'genre_War', 'genre_Western', 'language_English', 'language_French', 'language_Hindi', 'language_Mandarin', 'language_Other', 'language_Spanish', 'country_Australia', 'country_Canada', 'country_China', 'country_France', 'country_Germany', 'country_India', 'country_Italy', 'country_Japan', 'country_Other', 'country_Spain', 'country_UK', 'country_USA', 'content_rating_Adult', 'content_rating_Explicit', 'content_rating_Kids', 'content_rating_Other', 'content_rating_Teen', 'content_rating_Young']
    missing_cols = set(train_columns) - set(df.columns)
    for col in missing_cols:
        # Fill missing columns with zeros (meaning the genre doesn't exist in this movie)
        df[col] = 0

    # Ensure column order matches the training data (important for the model)
    # Reorder columns to match the training data exactly
    df = df[train_columns]

    return df
