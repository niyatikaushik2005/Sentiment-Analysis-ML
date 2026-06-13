from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}


class TextInput(BaseModel):
    text: str
@app.post("/predict")
def predict(input: TextInput):
    result = classifier(input.text)[0]  
    label = result["label"]            
    score = result["score"]             
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    return {
            "sentiment": label_map[label],
            "confidence": round(score, 4)
            }
    