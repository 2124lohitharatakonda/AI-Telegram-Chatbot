"""
NLP Engine — Intent Classification & Entity Extraction
Supports multi-turn conversation context with TF-IDF and cosine similarity.
"""

import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

MODEL_PATH = "models/nlp_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# ---------------------------------------------------------------------------
# Intent Training Data
# ---------------------------------------------------------------------------

INTENT_DATA = {
    "greet": [
        "hello", "hi", "hey", "good morning", "good evening",
        "hi there", "hello bot", "howdy", "greetings", "what's up",
    ],
    "faq_query": [
        "what is the refund policy", "how do I return an item",
        "what are your working hours", "where is your office",
        "how to contact support", "what payment methods do you accept",
        "is there a warranty", "what is the delivery time",
    ],
    "doc_search": [
        "find the manual for", "search document", "get the pdf for",
        "find user guide", "where is the document", "retrieve file",
        "show me the report", "look up the specification",
    ],
    "small_talk": [
        "how are you", "what can you do", "are you a bot",
        "tell me a joke", "what is your name", "who made you",
        "what do you know", "are you smart",
    ],
    "escalate": [
        "talk to a human", "connect me to an agent", "I need help urgently",
        "this is not working", "I want to speak to someone",
        "your answers are wrong", "escalate my issue",
    ],
    "goodbye": [
        "bye", "goodbye", "see you", "thanks bye", "exit",
        "quit", "that's all", "done for now",
    ],
}

ENTITY_PATTERNS = {
    "product_model": r"\b([A-Z][0-9]{3,4}|[A-Z]{2,}\s?[0-9]+)\b",
    "order_id":      r"\b(ORD|ORDER)-?[0-9]{5,8}\b",
    "email":         r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone":         r"\b[6-9]\d{9}\b",
    "date":          r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_entities(text: str) -> dict:
    entities = {}
    for entity_type, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            entities[entity_type] = matches
    return entities


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

def build_dataset():
    texts, labels = [], []
    for intent, examples in INTENT_DATA.items():
        for ex in examples:
            texts.append(preprocess(ex))
            labels.append(intent)
    return texts, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
        )),
    ])


def train() -> tuple:
    texts, labels = build_dataset()
    le = LabelEncoder()
    y = le.fit_transform(labels)

    model = build_pipeline()
    model.fit(texts, y)

    import os; os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    y_pred = model.predict(texts)
    print("=== NLP Model Training Complete ===")
    print(classification_report(y, y_pred, target_names=le.classes_))
    return model, le


def load_model():
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_intent(text: str, model, le) -> dict:
    clean = preprocess(text)
    probs = model.predict_proba([clean])[0]
    top_idx = np.argmax(probs)
    confidence = float(probs[top_idx])
    intent = le.inverse_transform([top_idx])[0]

    if confidence < 0.35:
        intent = "fallback"

    return {
        "intent":     intent,
        "confidence": round(confidence, 4),
        "entities":   extract_entities(text),
        "all_scores": {
            le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }


# ---------------------------------------------------------------------------
# Context Manager (multi-turn)
# ---------------------------------------------------------------------------

class ConversationContext:
    def __init__(self, max_turns: int = 5):
        self.history = []
        self.max_turns = max_turns
        self.entities = {}

    def add_turn(self, role: str, text: str, intent: str = None):
        self.history.append({"role": role, "text": text, "intent": intent})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def update_entities(self, entities: dict):
        self.entities.update(entities)

    def get_context_summary(self) -> str:
        recent = self.history[-4:]
        return " ".join(t["text"] for t in recent if t["role"] == "user")

    def last_intent(self) -> str:
        for turn in reversed(self.history):
            if turn["role"] == "user" and turn.get("intent"):
                return turn["intent"]
        return None

    def reset(self):
        self.history.clear()
        self.entities.clear()


if __name__ == "__main__":
    model, le = train()
    ctx = ConversationContext()

    test_inputs = [
        "hello there",
        "what is your refund policy",
        "what if the item is damaged",
        "find the manual for X200",
        "talk to a human agent",
        "bye",
    ]

    print("\n=== Intent Classification Demo ===")
    for text in test_inputs:
        result = classify_intent(text, model, le)
        ctx.add_turn("user", text, result["intent"])
        print(f"\nInput    : {text}")
        print(f"Intent   : {result['intent']}  ({result['confidence']*100:.1f}%)")
        if result["entities"]:
            print(f"Entities : {result['entities']}")
