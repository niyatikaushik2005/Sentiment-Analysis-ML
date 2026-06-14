import gradio as gr
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
def predict_sentiment(text):
    result = classifier(text)[0]
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    sentiment = label_map[result["label"]]
    confidence = round(result["score"], 4)
    return f"{sentiment} (confidence: {confidence})"

demo = gr.Interface(
    fn=predict_sentiment,          
    inputs="text",          
    outputs="text",         
    title="Tweet Sentiment: Classical ML vs RoBERTa"          
)

demo.launch()