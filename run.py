"""
Main launcher script for the AI-Powered Chatbot.
Usage:
    python run.py             # Start the web server (default)
    python run.py --cli       # Start terminal chat session
    python run.py --test      # Run automated tests
"""

import argparse
import sys
import uuid


def run_cli():
    """Runs an interactive conversational session in the terminal."""
    from dialog_manager import DialogManager
    from nlp_engine import NLPEngine
    import database

    print("=" * 60)
    print(" 🤖 Welcome to Nova AI Customer Support (Terminal Mode)")
    print(" Type 'quit', 'exit', or 'reset' anytime.")
    print(" Example queries:")
    print("   - 'Where is my order?'")
    print("   - 'Track order ORD-1001'")
    print("   - 'What is your return policy?'")
    print("   - 'I want to speak with a human'")
    print("=" * 60)

    database.init_db()
    nlp = NLPEngine()
    dm = DialogManager(nlp)
    session_id = f"cli_{uuid.uuid4().hex[:8]}"

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("Nova: Thank you for chatting! Have a great day.")
                break

            result = dm.process_message(session_id, user_input)
            print(f"\nNova: {result['response']}")
            print(
                f"[Diagnostics] Intent: {result['intent']} | "
                f"Confidence: {int(result['confidence'] * 100)}% | "
                f"State: {result['state']} | "
                f"Latency: {result['latency_ms']}ms"
            )

            if result.get("card"):
                card = result["card"]
                print(f"[Card Data - {card.get('type')}] {card}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break


def run_server(port=5000, host="0.0.0.0"):
    """Starts the Flask web application."""
    from app import app

    print("=" * 60)
    print(f" 🚀 AI-Powered Chatbot Web Server")
    print(f" Chat UI:     http://localhost:{port}/")
    print(f" Admin Logs:  http://localhost:{port}/admin")
    print(f" API Docs:    POST http://localhost:{port}/api/chat")
    print("=" * 60)
    app.run(host=host, port=port, debug=False)


def run_tests():
    """Executes the unit and integration test suite."""
    import unittest

    suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered ChatBot Runner")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI chat")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")

    args = parser.parse_args()

    if args.cli:
        run_cli()
    elif args.test:
        run_tests()
    else:
        run_server(port=args.port, host=args.host)
