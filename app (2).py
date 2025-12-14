import torch
import json
import gradio as gr
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Config
from torch import nn
from scipy.special import expit

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "microsoft/deberta-v3-large"
NUM_LABELS = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

# -------------------------
# MODEL DEFINITION
# -------------------------
class DebertaForMultiLabel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = DebertaV2Config.from_pretrained(model_name)
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return logits


# -------------------------
# LOAD MODEL & TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = DebertaForMultiLabel(MODEL_NAME, NUM_LABELS)
model.load_state_dict(torch.load("deberta_v3_best_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# LOAD THRESHOLDS
# -------------------------
with open("best_thresholds.json", "r") as f:
    best_thresholds = json.load(f)

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_emotions(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    probs = expit(logits.cpu().numpy()[0])

    results = []
    for i, label in enumerate(labels):
        if probs[i] > best_thresholds[i]:
            results.append(f"{label}: {probs[i]*100:.2f}%")

    if not results:
        results.append("No strong emotion detected")

    return "\n".join(results)

# -------------------------
# GRADIO UI
# -------------------------
demo = gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(lines=4, placeholder="Type a sentence..."),
    outputs=gr.Textbox(),
    title="Multilabel Emotion Detection (DeBERTa)",
    description="Fine-tuned DeBERTa-v3-large on GoEmotions with threshold optimization"
)

demo.launch()
