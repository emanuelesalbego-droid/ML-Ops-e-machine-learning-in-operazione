import pytest
from src.predict import predict_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I love this product, it is amazing!", "POS"),
        ("This is okay, not great but not bad.", "NEU"),
        ("I hate it, the service was terrible.", "NEG"),
    ],
)
def test_sentiment_prediction_contains_label(text, expected):
    result = predict_text(text)
    assert isinstance(result, list)
    assert len(result) == 1
    assert "label" in result[0]
    assert "score" in result[0]
    assert result[0]["label"] in {
        "positive",
        "negative",
        "neutral",
        "POSITIVE",
        "NEGATIVE",
        "NEUTRAL",
        "LABEL_0",
        "LABEL_1",
        "LABEL_2",
    }
