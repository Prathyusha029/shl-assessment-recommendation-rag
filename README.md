# shl-assessment-recommendation-rag
# SHL Assessment Recommendation Engine (RAG-based)

## Overview
This project builds a web-based Retrieval-Augmented Recommendation system using SHL's product catalog.

## Features
- Scrapes and stores SHL product catalog
- TF-IDF based retrieval
- Domain-aware boosting
- FastAPI backend
- JSON API endpoint
- Evaluation using Recall@10

## API Usage

Endpoint:
POST /recommend

Example request:
{
  "query": "Java developer with communication skills"
}

Returns:
{
  "recommendations": ["url1", "url2", ...]
}

## Run Locally

pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000

Then open:
http://127.0.0.1:8000/docs

## Evaluation

Baseline TF-IDF Recall@10: 0.155  
Hybrid Boosted Recall@10: 0.216  

## Author
Prathyusha Yekamba
