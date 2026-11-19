# 🤖 AI-Powered Customer Support & FAQ Chatbot

An intelligent, high-speed customer support chatbot built with **Natural Language Processing (NLP)**, **Flask**, and **SQLite**. Designed for fast contextual multi-turn dialogs, automatic entity extraction, sentiment-aware escalations, and interaction logging with an admin telemetry dashboard.

---

## 🌟 Key Features

- **⚡ Lightning-Fast NLP Engine (< 15ms latency):**
  - **NLTK Preprocessing:** Tokenization, lemmatization (`WordNetLemmatizer`), punctuation normalization, and negation-preserving stopword filtering.
  - **TF-IDF Semantic Matching:** Vector space representation with sublinear TF and n-gram analysis (unigrams & bigrams) for intent classification and FAQ matching.
  - **Transformers Integration:** Subword tokenization and sequence utilities compatible with Hugging Face `transformers`.
  - **Entity Extraction:** Automatic regex-based extraction of Order IDs (`ORD-1001`, `#1002`), emails, phone numbers, and amounts.
  - **Sentiment Detection:** Lexical sentiment analysis to detect frustrated customers and proactively route to human support.

- **🔄 Multi-Turn Context Management:**
  - Maintains conversation state across sessions in SQLite.
  - Contextual workflows: Order tracking status checks, return label authorization generation, and support ticket creation.

- **📊 SQLite Interaction Logging & Telemetry:**
  - Automatically records every user interaction (`session_id`, `user_message`, `bot_response`, `intent`, `confidence`, `sentiment`, `latency_ms`, `timestamp`).
  - Dedicated `/admin` dashboard to inspect live logs, query analytics, and monitor bot performance.

- **🎨 Modern Responsive Chat UI:**
  - Clean web interface with smooth message bubbles, typing indicator, and quick-reply suggestion chips.
  - Rich interactive widgets: Order status cards, Return authorization labels, and Support ticket badges.
  - Built-in live **NLP Diagnostic Inspector** to see real-time intent, confidence, sentiment, and response time.

---

## 🏗️ Architecture & Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core application logic |
| **NLP & Text Processing** | NLTK & Transformers | Tokenization, lemmatization, subwords, stopwords |
| **Semantic Matching** | Scikit-learn (TF-IDF) & NumPy | High-speed intent classification & FAQ cosine similarity |
| **Backend & REST API** | Flask | Web routes and JSON API endpoints |
| **Database** | SQLite3 | Interaction logs, sessions, mock orders, and tickets |
| **Frontend** | HTML5, CSS3, Vanilla JS | Glassmorphic responsive interface with FontAwesome icons |

---

## 📁 Project Structure

```
AI-Powered-ChatBot/
├── app.py                     # Flask application and RESTful API endpoints
├── nlp_engine.py              # NLP, NLTK preprocessing, TF-IDF matcher, Transformers tokenizer
├── dialog_manager.py          # Multi-turn context manager and state routing
├── database.py                # SQLite schema, queries, interaction logging, and order lookup
├── run.py                     # CLI launcher (Web Server, Terminal Chat, Unit Tests)
├── requirements.txt           # Project dependencies
├── data/
│   ├── intents.json           # FAQ patterns, responses, and contextual intents
│   └── chatbot.db             # SQLite database (auto-generated with seed orders)
├── static/
│   ├── css/
│   │   └── style.css          # Modern dark/glassmorphic responsive styling
│   └── js/
│       └── chat.js            # Client-side messaging, card widgets, and diagnostics
├── templates/
│   ├── index.html             # Customer support chat web interface
│   └── admin.html             # Interaction logs and analytics dashboard
└── tests/
    └── test_chatbot.py        # Automated unit and integration test suite
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Open the Project
```bash
cd AI-Powered-ChatBot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required NLTK Corpora (One-time)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 💻 Running the Chatbot

### Option A: Launch Web Server (Recommended)
```bash
python run.py
# or
python app.py
```
- Open your browser at: **[http://localhost:5000](http://localhost:5000)**
- Admin Analytics & Logs: **[http://localhost:5000/admin](http://localhost:5000/admin)**

### Option B: Interactive Terminal Mode (CLI)
Test the chatbot directly from your command prompt:
```bash
python run.py --cli
```

### Option C: Run Automated Test Suite
Run the 10 automated unit and integration tests:
```bash
python run.py --test
# or
python -m unittest tests/test_chatbot.py
```

---

## 🧪 Demo Test Scenarios

Try these queries in the chat or CLI:

| Category | Example User Query | Expected Behavior |
|---|---|---|
| **Order Tracking (Multi-turn)** | *"Where is my order?"* → then reply *"ORD-1001"* | Bot prompts for ID, retrieves order from SQLite, and renders an Order Status card. |
| **Direct Order Tracking** | *"Track order ORD-1002"* | Extracts ID inline, checks database, and displays delivery info. |
| **Returns & Refunds** | *"How do I return an item?"* → *"ORD-1003"* | Guides return policy and generates a valid Return Authorization RMA code. |
| **Human Escalation** | *"I want to speak with a human"* → *"open a ticket"* | Prompts for support choice and creates a tracked support ticket in SQLite (`TCK-XXXX`). |
| **FAQs** | *"What are your shipping rates?"* or *"What payment methods do you accept?"* | Provides instant, clear policy details. |
| **Reset State** | Click **Reset** button or type *"reset"* | Clears session context back to clean slate. |

### Seeded Mock Orders in SQLite:
- `ORD-1001`: In Transit via FedEx Express (Headphones)
- `ORD-1002`: Delivered via UPS Ground (Mechanical Keyboard)
- `ORD-1003`: Processing via USPS Priority (4K Gaming Monitor)
- `ORD-1004`: Out for Delivery via DHL Express (Smart Fitness Watch)
- `ORD-1005`: Return Pending (Portable Speaker)

---

## 📡 REST API Reference

### 1. Send Message
- **Endpoint:** `POST /api/chat`
- **Request Body:**
  ```json
  {
    "message": "Where is my order ORD-1001?",
    "session_id": "sess_demo_123"
  }
  ```
- **Response:**
  ```json
  {
    "success": true,
    "session_id": "sess_demo_123",
    "log_id": 4,
    "response": "Found your order **ORD-1001**!...",
    "intent": "track_order",
    "confidence": 1.0,
    "sentiment": "neutral",
    "state": "default",
    "latency_ms": 11.45,
    "card": {
      "type": "order_status",
      "order_id": "ORD-1001",
      "status": "In Transit",
      "carrier": "FedEx Express"
    }
  }
  ```

### 2. View Interaction Logs
- **Endpoint:** `GET /api/logs?limit=50&offset=0`
- **Response:** Returns paginated list of logged user interactions and total count.

### 3. View Analytics Stats
- **Endpoint:** `GET /api/stats`
- **Response:** Summary metrics (total interactions, unique sessions, average confidence, average latency).

### 4. Reset Context
- **Endpoint:** `POST /api/reset`
  ```json
  { "session_id": "sess_demo_123" }
  ```

### 5. Submit Message Feedback
- **Endpoint:** `POST /api/feedback`
  ```json
  { "log_id": 4, "rating": "positive", "comment": "Very helpful!" }
  ```

---

## 📄 License
MIT License. Built for Codec Tech Internship Assessment.
