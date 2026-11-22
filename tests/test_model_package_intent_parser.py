import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_out"))

from intent_parser import parse_intent


def test_hyphenated_query_identifies_core_details_and_premium_preference():
    intent = parse_intent(
        "I am 95 years old and looking for a low-premium policy with "
        "5 lakh sum assured for a 10-year term"
    )

    assert intent["customer_age"] == 95
    assert intent["sum_assured"] == 500_000
    assert intent["term"] == 10
    assert intent["premium_level"] == "Low"
