import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import preprocessing 
import uvicorn


MODEL_PATH = "models/model.bin"
PROCESSED_HEADER = "data/processed/movie_metadata_processed.csv"

def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

class MovieInput(BaseModel):
    budget: float
    gross: float
    duration: float
    num_critic_for_reviews: int
    num_user_for_reviews: int
    num_voted_users: int
    title_year: int
    color: bool = Field(..., description="Colored?")
    genres: str = Field(
        ...,
        description="Pipe-separated list of genres for the movie. E.g. 'Action|Comedy|Drama'. "
                    "Allowed genres are: Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, "
                    "Family, Fantasy, Film-Noir, Game-Show, History, Horror, Music, Musical, Mystery, News, Reality-TV, "
                    "Romance, Sci-Fi, Short, Sport, Thriller, War, Western.",
        example="Action|Comedy|Drama"
    )
    language: Literal['English', 'French', 'Spanish', 'Hindi', 'Mandarin', 'Other'] = Field(..., description="The language of the movie.")
    country: Literal['USA', 'UK', 'France', 'India', 'Australia', 'Other'] = Field(..., description="The country where the movie was produced.")
    content_rating: Literal['G', 'PG', 'PG-13', 'R', 'NC-17', 'Unrated'] = Field(..., description="The content rating of the movie.")

class PredictResponse(BaseModel):
    is_good_probability: float
    is_good: bool

app = FastAPI(title="movie-quality-prediction")

model = load_model()

def preprocess_input(input_data: MovieInput):
    data = pd.DataFrame([input_data.dict()])
    processed_data = preprocessing.preprocess_for_inference(data)
    return processed_data

@app.post("/predict", response_model=PredictResponse)
async def predict(movie: MovieInput):
    preprocessed_data = preprocess_input(movie)
    
    prediction_probability = model.predict_proba(preprocessed_data)[0, 1]  # Probability for class 1 ("good")
    prediction_class = model.predict(preprocessed_data)[0]  # Final prediction class (0 or 1)
    
    return PredictResponse(is_good_probability=prediction_probability, is_good=bool(prediction_class))

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
