from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

catalog_df = pd.read_csv("catalog.csv")

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

catalog_df["clean_text"] = (
    catalog_df["name"].fillna("") + " " +
    catalog_df["url"].fillna("")
).apply(preprocess)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(catalog_df["clean_text"])

def retrieve_top_10(query):
    query_clean = preprocess(query)
    query_vec = vectorizer.transform([query_clean])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-10:][::-1]
    return catalog_df.iloc[top_indices]["url"].tolist()

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend(request: QueryRequest):
    results = retrieve_top_10(request.query)
    return {"recommendations": results}