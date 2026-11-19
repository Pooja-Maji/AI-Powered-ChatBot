"""
Automated Unit and Integration Tests for AI-Powered Chatbot.
Covers NLP Engine, Database layer, Dialog Management, and Flask API endpoints.
"""

import json
import os
import sys
import unittest

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
from app import app
from dialog_manager import DialogManager
from nlp_engine import NLPEngine


class TestChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()
        cls.nlp = NLPEngine()
        cls.dm = DialogManager(cls.nlp)
        cls.client = app.test_client()

    def test_01_nlp_preprocessing(self):
        """Verifies NLTK tokenization, lemmatization and cleaning."""
        cleaned = self.nlp.preprocess("Where is my orders and packages???")
        self.assertIn("order", cleaned)
        self.assertNotIn("???", cleaned)

    def test_02_intent_prediction(self):
        """Verifies intent classification across FAQ topics."""
        intent, conf = self.nlp.predict_intent("Hello there!")
        self.assertEqual(intent, "greeting")
        self.assertGreater(conf, 0.4)

        intent, conf = self.nlp.predict_intent("track my order")
        self.assertEqual(intent, "track_order")
        self.assertGreater(conf, 0.5)

        intent, conf = self.nlp.predict_intent("what is your return policy?")
        self.assertEqual(intent, "returns_refunds")
        self.assertGreater(conf, 0.4)

        intent, conf = self.nlp.predict_intent("can i speak to a human representative?")
        self.assertEqual(intent, "human_escalation")
        self.assertGreater(conf, 0.4)

    def test_03_entity_extraction(self):
        """Verifies regex entity extraction for order IDs and emails."""
        entities = self.nlp.extract_entities("Please check order ORD-1002 for john@example.com")
        self.assertEqual(entities.get("order_id"), "ORD-1002")
        self.assertEqual(entities.get("email"), "john@example.com")

        # Numeric order fallback
        entities2 = self.nlp.extract_entities("my order is #1003")
        self.assertEqual(entities2.get("order_id"), "ORD-1003")

    def test_04_sentiment_analysis(self):
        """Verifies lexical sentiment detection."""
        s_pos, score_pos = self.nlp.analyze_sentiment("Thank you so much, this is awesome and great!")
        self.assertEqual(s_pos, "positive")
        self.assertGreater(score_pos, 0.0)

        s_neg, score_neg = self.nlp.analyze_sentiment("This service is terrible, horrible and completely broken!")
        self.assertEqual(s_neg, "negative")
        self.assertLess(score_neg, 0.0)

    def test_05_database_order_lookup(self):
        """Verifies querying SQLite mock order knowledge base."""
        order = database.get_order("ORD-1001")
        self.assertIsNotNone(order)
        self.assertEqual(order["order_id"], "ORD-1001")
        self.assertEqual(order["customer_name"], "Alex Johnson")
        self.assertIn("FedEx", order["carrier"])

    def test_06_contextual_multi_turn_order_flow(self):
        """Tests multi-turn context: asking to track order, then supplying ID in turn 2."""
        session_id = "test_turn_session_100"

        # Turn 1: User asks to track order without providing ID
        r1 = self.dm.process_message(session_id, "I want to track my order")
        self.assertEqual(r1["intent"], "track_order")
        self.assertEqual(r1["state"], "awaiting_order_id")
        self.assertIsNone(r1["card"])

        # Turn 2: User provides Order ID
        r2 = self.dm.process_message(session_id, "my order id is ORD-1001")
        self.assertEqual(r2["state"], "default")
        self.assertIsNotNone(r2["card"])
        self.assertEqual(r2["card"]["order_id"], "ORD-1001")
        self.assertEqual(r2["card"]["status"], "In Transit")

    def test_07_direct_order_tracking(self):
        """Tests single-turn order tracking when ID is provided in query."""
        session_id = "test_direct_session_200"
        r = self.dm.process_message(session_id, "Where is order ORD-1002?")
        self.assertEqual(r["intent"], "track_order")
        self.assertIsNotNone(r["card"])
        self.assertEqual(r["card"]["order_id"], "ORD-1002")
        self.assertEqual(r["card"]["status"], "Delivered")

    def test_08_human_escalation_flow(self):
        """Tests human escalation creating a support ticket in SQLite."""
        session_id = "test_escalation_session_300"
        r1 = self.dm.process_message(session_id, "I want to speak with a human")
        self.assertEqual(r1["state"], "awaiting_escalation_choice")

        # User chooses to create a ticket
        r2 = self.dm.process_message(session_id, "Please open a support ticket")
        self.assertEqual(r2["state"], "default")
        self.assertIsNotNone(r2["card"])
        self.assertEqual(r2["card"]["type"], "support_ticket")
        self.assertTrue(r2["card"]["ticket_id"].startswith("TCK-"))

    def test_09_flask_chat_endpoint(self):
        """Verifies Flask REST API /api/chat endpoint."""
        res = self.client.post(
            "/api/chat",
            data=json.dumps({"message": "Hello, who are you?", "session_id": "test_api_sess"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertIn("Nova", data["response"])
        self.assertGreater(data["latency_ms"], 0)

    def test_10_flask_logs_and_stats_endpoints(self):
        """Verifies /api/logs and /api/stats endpoints."""
        res_logs = self.client.get("/api/logs?limit=5")
        self.assertEqual(res_logs.status_code, 200)
        data_logs = json.loads(res_logs.data)
        self.assertTrue(data_logs["success"])
        self.assertGreater(data_logs["total"], 0)

        res_stats = self.client.get("/api/stats")
        self.assertEqual(res_stats.status_code, 200)
        data_stats = json.loads(res_stats.data)
        self.assertTrue(data_stats["success"])
        self.assertIn("total_logs", data_stats["stats"])


if __name__ == "__main__":
    unittest.main()
