"""
Database module for AI-Powered Chatbot.
Handles SQLite schema creation, interaction logging, session persistence,
order queries, and analytics.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "chatbot.db")


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection configured for dictionaries."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database tables and seeds demo orders."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. User Interaction Logs
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                intent TEXT,
                confidence REAL,
                sentiment TEXT,
                context_state TEXT,
                latency_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # 2. Conversational Sessions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                state TEXT DEFAULT 'default',
                context_json TEXT DEFAULT '{}',
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # 3. Mock Orders Knowledge Base for Customer Support
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_name TEXT NOT NULL,
                status TEXT NOT NULL,
                carrier TEXT,
                tracking_code TEXT,
                items TEXT,
                estimated_delivery TEXT,
                total_price TEXT
            )
        """
        )

        # 4. Support Tickets (for human escalation)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_message TEXT,
                priority TEXT DEFAULT 'NORMAL',
                status TEXT DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # 5. User Feedback (helpful / unhelpful)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER,
                rating TEXT,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Seed mock orders if empty
        cursor.execute("SELECT COUNT(*) as count FROM orders")
        if cursor.fetchone()["count"] == 0:
            mock_orders = [
                (
                    "ORD-1001",
                    "Alex Johnson",
                    "In Transit",
                    "FedEx Express",
                    "FX-982341209",
                    "Wireless Noise-Canceling Headphones (Black)",
                    "Tomorrow by 5:00 PM",
                    "$149.99",
                ),
                (
                    "ORD-1002",
                    "Samantha Reed",
                    "Delivered",
                    "UPS Ground",
                    "1Z9999999999999999",
                    "Ergonomic Mechanical Keyboard (RGB)",
                    "Delivered yesterday (Signed by front desk)",
                    "$89.50",
                ),
                (
                    "ORD-1003",
                    "Michael Chen",
                    "Processing",
                    "USPS Priority",
                    "US-450123891",
                    "Ultra-Wide 4K Gaming Monitor (34-inch)",
                    "Estimated dispatch in 24 hours",
                    "$429.00",
                ),
                (
                    "ORD-1004",
                    "Emma Davis",
                    "Out for Delivery",
                    "DHL Express",
                    "DHL-33290145",
                    "Smart Fitness Watch v3 (Silver)",
                    "Today before 7:00 PM",
                    "$199.95",
                ),
                (
                    "ORD-1005",
                    "David Miller",
                    "Return Pending",
                    "Return Carrier",
                    "RET-774012",
                    "Bluetooth Portable Speaker",
                    "Awaiting item arrival at warehouse",
                    "$59.00",
                ),
            ]
            cursor.executemany(
                """
                INSERT INTO orders (order_id, customer_name, status, carrier, tracking_code, items, estimated_delivery, total_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                mock_orders,
            )

        conn.commit()


def log_interaction(
    session_id: str,
    user_message: str,
    bot_response: str,
    intent: str,
    confidence: float,
    sentiment: str = "neutral",
    context_state: str = "default",
    latency_ms: float = 0.0,
) -> int:
    """Logs an interaction turn to SQLite."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO interactions (
                session_id, user_message, bot_response, intent,
                confidence, sentiment, context_state, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                user_message,
                bot_response,
                intent,
                confidence,
                sentiment,
                context_state,
                latency_ms,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieves session state and context dict from database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT state, context_json FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            try:
                context = json.loads(row["context_json"])
            except Exception:
                context = {}
            return {"session_id": session_id, "state": row["state"], "context": context}
        else:
            # Create session
            cursor.execute(
                "INSERT INTO sessions (session_id, state, context_json) VALUES (?, 'default', '{}')",
                (session_id,),
            )
            conn.commit()
            return {"session_id": session_id, "state": "default", "context": {}}


def save_session(session_id: str, state: str, context: Dict[str, Any]):
    """Persists session state and context dict."""
    with get_connection() as conn:
        cursor = conn.cursor()
        context_str = json.dumps(context)
        cursor.execute(
            """
            INSERT INTO sessions (session_id, state, context_json, last_active)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                state = excluded.state,
                context_json = excluded.context_json,
                last_active = CURRENT_TIMESTAMP
        """,
            (session_id, state, context_str),
        )
        conn.commit()


def reset_session(session_id: str):
    """Resets session state to default."""
    save_session(session_id, "default", {})


def get_order(order_id: str) -> Optional[Dict[str, Any]]:
    """Look up an order by ID (case-insensitive)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM orders WHERE UPPER(order_id) = UPPER(?)", (order_id.strip(),)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def create_support_ticket(
    session_id: str, message: str, priority: str = "NORMAL"
) -> str:
    """Creates a support ticket for human escalation."""
    import uuid

    ticket_id = f"TCK-{datetime.now().strftime('%Y%m')}-{uuid.uuid4().hex[:4].upper()}"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO support_tickets (ticket_id, session_id, user_message, priority)
            VALUES (?, ?, ?, ?)
        """,
            (ticket_id, session_id, message, priority),
        )
        conn.commit()
    return ticket_id


def record_feedback(
    interaction_id: Optional[int], rating: str, comment: str = ""
) -> int:
    """Stores user feedback for bot responses."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (interaction_id, rating, comment) VALUES (?, ?, ?)",
            (interaction_id, rating, comment),
        )
        conn.commit()
        return cursor.lastrowid


def get_interaction_logs(
    limit: int = 50, offset: int = 0, session_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """Retrieves paginated interaction logs with total count."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if session_id:
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM interactions WHERE session_id = ?",
                (session_id,),
            )
            total = cursor.fetchone()["cnt"]
            cursor.execute(
                """
                SELECT * FROM interactions
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
                (session_id, limit, offset),
            )
        else:
            cursor.execute("SELECT COUNT(*) as cnt FROM interactions")
            total = cursor.fetchone()["cnt"]
            cursor.execute(
                """
                SELECT * FROM interactions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

        rows = [dict(r) for r in cursor.fetchall()]
        return rows, total


def get_analytics_summary() -> Dict[str, Any]:
    """Returns analytics metrics for the admin dashboard."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total_logs FROM interactions")
        total_logs = cursor.fetchone()["total_logs"]

        cursor.execute("SELECT COUNT(DISTINCT session_id) as total_sessions FROM sessions")
        total_sessions = cursor.fetchone()["total_sessions"]

        cursor.execute("SELECT AVG(confidence) as avg_conf, AVG(latency_ms) as avg_lat FROM interactions")
        row = cursor.fetchone()
        avg_confidence = round(row["avg_conf"] or 0.0, 3)
        avg_latency = round(row["avg_lat"] or 0.0, 2)

        # Intent distribution
        cursor.execute(
            """
            SELECT intent, COUNT(*) as count
            FROM interactions
            GROUP BY intent
            ORDER BY count DESC
            LIMIT 6
        """
        )
        intent_stats = [dict(r) for r in cursor.fetchall()]

        # Sentiment distribution
        cursor.execute(
            """
            SELECT sentiment, COUNT(*) as count
            FROM interactions
            GROUP BY sentiment
        """
        )
        sentiment_stats = {r["sentiment"]: r["count"] for r in cursor.fetchall()}

        # Open support tickets
        cursor.execute("SELECT COUNT(*) as open_tickets FROM support_tickets WHERE status = 'OPEN'")
        open_tickets = cursor.fetchone()["open_tickets"]

        return {
            "total_logs": total_logs,
            "total_sessions": total_sessions,
            "avg_confidence": avg_confidence,
            "avg_latency_ms": avg_latency,
            "intent_stats": intent_stats,
            "sentiment_stats": sentiment_stats,
            "open_tickets": open_tickets,
        }
