"""
Flask Application for AI-Powered Customer Support & FAQ Chatbot.
Provides RESTful endpoints for chat interactions, analytics,
interaction logs, and web interfaces.
"""

import logging
import os
import uuid
from flask import Flask, jsonify, render_template, request

import database
from dialog_manager import DialogManager
from nlp_engine import NLPEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "ai-chatbot-secret-key-2026"

# Initialize Database and NLP components
database.init_db()
nlp = NLPEngine()
dialog_mgr = DialogManager(nlp)


@app.route("/")
def index():
    """Renders the Customer Support Chatbot UI."""
    return render_template("index.html")


@app.route("/admin")
def admin():
    """Renders the Admin & Interaction Logs Analytics Dashboard."""
    return render_template("admin.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main Chat API Endpoint.
    Payload: {"message": str, "session_id": Optional[str]}
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    if not session_id:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

    try:
        result = dialog_mgr.process_message(session_id, user_message)
        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "log_id": result["log_id"],
                "response": result["response"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "sentiment": result["sentiment"],
                "sentiment_score": result["sentiment_score"],
                "state": result["state"],
                "card": result["card"],
                "latency_ms": result["latency_ms"],
            }
        )
    except Exception as e:
        logger.exception("Error processing chat message:")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    """Resets conversational context for a given session."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()

    if session_id:
        database.reset_session(session_id)

    return jsonify({"success": True, "message": "Context reset successfully."})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns paginated user interaction logs from SQLite."""
    limit = min(int(request.args.get("limit", 50)), 100)
    offset = max(int(request.args.get("offset", 0)), 0)
    session_id = request.args.get("session_id", None)

    logs, total = database.get_interaction_logs(
        limit=limit, offset=offset, session_id=session_id
    )
    return jsonify({"success": True, "total": total, "logs": logs})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Returns analytics metrics for dashboard."""
    stats = database.get_analytics_summary()
    return jsonify({"success": True, "stats": stats})


@app.route("/api/feedback", methods=["POST"])
def post_feedback():
    """Records user feedback on bot answers."""
    data = request.get_json(silent=True) or {}
    log_id = data.get("log_id")
    rating = data.get("rating", "positive")
    comment = data.get("comment", "")

    feedback_id = database.record_feedback(log_id, rating, comment)
    return jsonify({"success": True, "feedback_id": feedback_id})


@app.route("/api/faqs", methods=["GET"])
def get_faqs():
    """Returns customer support FAQ categories and quick suggestions."""
    faqs = [
        {"intent": "track_order", "query": "Where is my order?"},
        {"intent": "shipping_policy", "query": "What are your shipping rates and times?"},
        {"intent": "returns_refunds", "query": "How do I return an item or get a refund?"},
        {"intent": "payment_methods", "query": "What payment methods do you accept?"},
        {"intent": "cancel_order", "query": "How can I cancel or modify my order?"},
        {"intent": "human_escalation", "query": "I want to speak with a human agent"},
    ]
    return jsonify({"success": True, "faqs": faqs})


@app.route("/api/tokenizer", methods=["POST"])
def subword_tokenization():
    """Transformers tokenization demonstration endpoint."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    token_data = nlp.tokenize_transformers(text)
    return jsonify({"success": True, "result": token_data})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AI Chatbot server on http://127.0.0.1:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
