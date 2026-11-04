"""
Flask app - poora flow:

  Browser (index.html + script.js)
        |  1. user message likhta hai, "Check message" dabata hai
        |  2. script.js -> POST /predict  {"message": "..."}
        v
  Flask (ye file)
        |  3. predict() message leta hai
        |  4. classify() -> preprocess -> TF-IDF -> Naive Bayes model
        |  5. jsonify(...) se result JSON me wapas
        v
  Browser: script.js result screen par dikhata hai
"""
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from model_utils import load_or_train
from preprocess import transform_text

app = Flask(__name__)

# App start hote hi saved model load hota hai (nahi mila to train hota hai)
tfidf, model, meta = load_or_train()

# Neeche ke 2 lines "kaunse words ne result ko influence kiya" nikalne ke kaam aate hain
VOCAB = tfidf.get_feature_names_out()
LOG_DIFF = model.feature_log_prob_[1] - model.feature_log_prob_[0]


def classify(message: str) -> dict:
    """Ek message lo -> spam/ham ka result dict do."""
    cleaned = transform_text(message)                  # lowercase, stopwords hatao, stemming
    vec = tfidf.transform([cleaned]).toarray()         # text -> numbers
    spam_p = float(model.predict_proba(vec)[0][1])     # spam hone ki probability (0 se 1)

    # Message me jo words the, unka weight nikalo (+ = spam, - = not spam)
    idx = vec[0].nonzero()[0]
    words = sorted(
        ({"word": str(VOCAB[i]), "weight": round(float(LOG_DIFF[i] * vec[0][i]), 3)} for i in idx),
        key=lambda w: -abs(w["weight"]),
    )[:8]

    return {
        "label": "spam" if spam_p >= 0.5 else "ham",
        "spam_probability": round(spam_p * 100, 2),
        "confidence": round(max(spam_p, 1 - spam_p) * 100, 2),
        "top_words": words,
        "processed_text": cleaned,
    }


# Route 1: homepage. templates/index.html dikhata hai
@app.route("/")
def index():
    return render_template("index.html", meta=meta)


# Route 2: prediction. script.js yaha POST karta hai
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 2000:
        return jsonify({"error": "Message too long (max 2000 characters)."}), 400
    return jsonify(classify(message))


# Koi bhi error aaye to JSON me wapas bhejo (taaki frontend usse dikha sake)
@app.errorhandler(Exception)
def handle_error(e):
    app.logger.exception(e)
    code = e.code if isinstance(e, HTTPException) else 500
    return jsonify({"error": f"Server error: {e}"}), code


if __name__ == "__main__":
    app.run(debug=False, port=5000)
