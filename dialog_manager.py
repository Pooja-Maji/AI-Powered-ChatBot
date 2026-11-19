"""
Contextual Dialog Manager for Customer Support & FAQs.
Manages multi-turn conversation states, slot filling, order lookups,
and human escalation workflows.
"""

import logging
import random
import time
from typing import Any, Dict, Optional, Tuple

import database
from nlp_engine import NLPEngine

logger = logging.getLogger(__name__)


class DialogManager:
    """
    Manages conversational dialog context, transitions, and business logic.
    """

    def __init__(self, nlp_engine: NLPEngine):
        self.nlp = nlp_engine

    def process_message(
        self, session_id: str, user_message: str
    ) -> Dict[str, Any]:
        """
        Main entry point for handling an incoming user message.
        Processes NLTK/TF-IDF intent, manages contextual state, queries SQLite,
        and logs the interaction.
        """
        start_time = time.time()
        user_message_clean = user_message.strip()

        # Retrieve current session state & context
        session_data = database.get_session(session_id)
        current_state = session_data.get("state", "default")
        context = session_data.get("context", {})

        # NLP predictions
        predicted_intent, confidence = self.nlp.predict_intent(user_message_clean)
        sentiment_label, sentiment_score = self.nlp.analyze_sentiment(
            user_message_clean
        )
        entities = self.nlp.extract_entities(user_message_clean)

        # Merge extracted entities into session context
        for key, val in entities.items():
            context[key] = val

        # Multi-turn context resolution
        response_text, new_state, card_data = self._route_dialog(
            user_message=user_message_clean,
            predicted_intent=predicted_intent,
            confidence=confidence,
            sentiment=sentiment_label,
            current_state=current_state,
            context=context,
            entities=entities,
            session_id=session_id,
        )

        # Persist updated session state
        database.save_session(session_id, new_state, context)

        # Calculate latency in ms
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Log interaction to SQLite database
        log_id = database.log_interaction(
            session_id=session_id,
            user_message=user_message_clean,
            bot_response=response_text,
            intent=predicted_intent,
            confidence=confidence,
            sentiment=sentiment_label,
            context_state=new_state,
            latency_ms=latency_ms,
        )

        return {
            "log_id": log_id,
            "session_id": session_id,
            "response": response_text,
            "intent": predicted_intent,
            "confidence": confidence,
            "sentiment": sentiment_label,
            "sentiment_score": sentiment_score,
            "state": new_state,
            "context": context,
            "card": card_data,
            "latency_ms": latency_ms,
        }

    def _route_dialog(
        self,
        user_message: str,
        predicted_intent: str,
        confidence: float,
        sentiment: str,
        current_state: str,
        context: Dict[str, Any],
        entities: Dict[str, Any],
        session_id: str,
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Routes the conversation based on state, intent, and sentiment.
        Returns: (response_text, next_state, optional_card_data)
        """
        user_lower = user_message.lower()

        # Check for reset / restart command
        if user_lower in ["reset", "restart", "start over", "clear"]:
            database.reset_session(session_id)
            context.clear()
            return (
                "Session context has been reset. How can I assist you today?",
                "default",
                None,
            )

        # State 1: Awaiting Order ID for Tracking
        if current_state == "awaiting_order_id":
            order_id = context.get("order_id")
            if not order_id:
                # Try to extract numbers from the message directly
                entities = self.nlp.extract_entities(user_message)
                order_id = entities.get("order_id")

            if order_id:
                order_info = database.get_order(order_id)
                if order_info:
                    card_data = {
                        "type": "order_status",
                        "order_id": order_info["order_id"],
                        "customer": order_info["customer_name"],
                        "status": order_info["status"],
                        "carrier": order_info["carrier"],
                        "tracking_code": order_info["tracking_code"],
                        "items": order_info["items"],
                        "delivery": order_info["estimated_delivery"],
                        "total": order_info["total_price"],
                    }
                    resp = (
                        f"Found your order **{order_info['order_id']}**! "
                        f"Current Status: **{order_info['status']}** with {order_info['carrier']}. "
                        f"Expected delivery: **{order_info['estimated_delivery']}**."
                    )
                    return resp, "default", card_data
                else:
                    return (
                        f"I couldn't find an order matching **{order_id}** in our system. "
                        "Please verify the number (Demo test orders: `ORD-1001`, `ORD-1002`, `ORD-1003`, `ORD-1004`, `ORD-1005`). "
                        "You can re-enter your order number, or type 'cancel' to exit.",
                        "awaiting_order_id",
                        None,
                    )
            elif "cancel" in user_lower or "nevermind" in user_lower:
                return "Order search cancelled. How else may I assist you?", "default", None
            else:
                return (
                    "Please provide a valid Order ID (e.g. `ORD-1001` or `1002`) so I can check its real-time shipping status for you.",
                    "awaiting_order_id",
                    None,
                )

        # State 2: Awaiting Return / Refund Details
        if current_state == "awaiting_return_details":
            order_id = context.get("order_id")
            if not order_id:
                entities = self.nlp.extract_entities(user_message)
                order_id = entities.get("order_id")
                if order_id:
                    context["order_id"] = order_id

            if order_id:
                rma_code = f"RMA-{random.randint(10000, 99999)}"
                card_data = {
                    "type": "return_auth",
                    "rma_code": rma_code,
                    "order_id": order_id,
                    "instructions": "A prepaid return label has been generated. Please print and attach to parcel.",
                }
                resp = (
                    f"Return approved for **{order_id}**! Your Return Authorization Code is **{rma_code}**. "
                    "We've sent a prepaid digital shipping label to the email address on file. "
                    "Once dropped off with the carrier, your refund will process in 2-3 business days."
                )
                return resp, "default", card_data
            else:
                return (
                    "To generate your free return label, please share your Order ID (e.g. `ORD-1001`).",
                    "awaiting_return_details",
                    None,
                )

        # State 3: Awaiting Escalation Choice (Human agent / Ticket)
        if current_state == "awaiting_escalation_choice":
            if any(w in user_lower for w in ["ticket", "yes", "email", "open", "create"]):
                ticket_id = database.create_support_ticket(
                    session_id=session_id,
                    message=user_message,
                    priority="HIGH" if sentiment == "negative" else "NORMAL",
                )
                card_data = {
                    "type": "support_ticket",
                    "ticket_id": ticket_id,
                    "status": "OPEN",
                    "priority": "HIGH" if sentiment == "negative" else "NORMAL",
                }
                resp = (
                    f"Support ticket **{ticket_id}** has been created with priority status! "
                    "A human support specialist has been assigned and will follow up shortly via email. "
                    "Is there anything else I can assist with in the meantime?"
                )
                return resp, "default", card_data
            elif any(w in user_lower for w in ["phone", "call"]):
                return (
                    "You can reach our live representative team toll-free at **1-800-555-0199** "
                    "(Mon-Fri 8:00 AM - 8:00 PM EST). Let me know if you'd also like a support ticket created!",
                    "default",
                    None,
                )
            else:
                return (
                    "Would you like me to open a support ticket for an agent (type 'ticket') or do you prefer our phone number?",
                    "awaiting_escalation_choice",
                    None,
                )

        # DEFAULT STATE: Intent matching & slot handling
        # 1. Proactive sentiment handling: If customer is expressing high frustration
        if sentiment == "negative" and confidence < 0.5:
            ticket_id = database.create_support_ticket(
                session_id=session_id,
                message=user_message,
                priority="HIGH",
            )
            card_data = {
                "type": "support_ticket",
                "ticket_id": ticket_id,
                "status": "OPEN",
                "priority": "HIGH",
            }
            return (
                f"I sincerely apologize for the frustration! I've automatically escalated this to our senior care team under Ticket **{ticket_id}**. "
                "A representative will review this immediately. Would you like our direct phone contact or more details?",
                "awaiting_escalation_choice",
                card_data,
            )

        # 2. Check if intent is order tracking and order ID was provided inline
        if predicted_intent == "track_order":
            order_id = entities.get("order_id")
            if order_id:
                order_info = database.get_order(order_id)
                if order_info:
                    card_data = {
                        "type": "order_status",
                        "order_id": order_info["order_id"],
                        "customer": order_info["customer_name"],
                        "status": order_info["status"],
                        "carrier": order_info["carrier"],
                        "tracking_code": order_info["tracking_code"],
                        "items": order_info["items"],
                        "delivery": order_info["estimated_delivery"],
                        "total": order_info["total_price"],
                    }
                    resp = (
                        f"Found your order **{order_info['order_id']}**! "
                        f"Status: **{order_info['status']}** via {order_info['carrier']}. "
                        f"Expected delivery: **{order_info['estimated_delivery']}**."
                    )
                    return resp, "default", card_data
                else:
                    return (
                        f"I checked for order **{order_id}**, but couldn't locate it in the system. "
                        "Please verify the number (Demo orders: `ORD-1001` through `ORD-1005`).",
                        "awaiting_order_id",
                        None,
                    )
            else:
                return (
                    "I'd be happy to track your package! What is your Order ID (e.g. `ORD-1001` or `1002`)?",
                    "awaiting_order_id",
                    None,
                )

        # 3. Check if intent is returns and order ID provided inline
        if predicted_intent == "returns_refunds":
            order_id = entities.get("order_id")
            if order_id:
                rma_code = f"RMA-{random.randint(10000, 99999)}"
                card_data = {
                    "type": "return_auth",
                    "rma_code": rma_code,
                    "order_id": order_id,
                }
                resp = (
                    f"Return initiated for **{order_id}**! Return Code: **{rma_code}**. "
                    "A return label has been emailed to you. Simply attach it to the box and drop it at any authorized drop-off station."
                )
                return resp, "default", card_data
            else:
                return (
                    "We offer a 30-day hassle-free return window! To proceed with a return or exchange, please provide your Order ID (e.g. `ORD-1001`).",
                    "awaiting_return_details",
                    None,
                )

        # 4. Human escalation intent
        if predicted_intent == "human_escalation":
            return (
                "I'm connecting you with our support resources! You can call us directly at **1-800-555-0199** "
                "or I can open an immediate support ticket for you right now. Type 'ticket' to create one.",
                "awaiting_escalation_choice",
                None,
            )

        # 5. Standard FAQ or Intent response
        bot_response = self.nlp.get_response_for_intent(predicted_intent)
        next_context = self.nlp.intent_context_map.get(predicted_intent, "")
        next_state = next_context if next_context else "default"

        return bot_response, next_state, None
