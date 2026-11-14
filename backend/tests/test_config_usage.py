from types import SimpleNamespace

import pytest

from app.config import ModelPrice, Settings
from app.llm.usage import cost_usd, usage_from_message

PRICES = {"m": ModelPrice(input=1.0, cached_input=0.5, output=2.0), "e": ModelPrice(input=0.1)}


def test_settings_from_env():
    s = Settings(_env_file=None)
    assert s.model_for("small") == "test-small"
    assert s.model_for("large") == "test-large"
    assert s.database_url.startswith("postgresql://")


def test_cost_with_cache():
    # 1000 de entrada, 400 vindos do cache, 500 de saída
    expected = (600 * 1.0 + 400 * 0.5 + 500 * 2.0) / 1e6
    assert cost_usd("m", 1000, 500, 400, PRICES) == pytest.approx(expected)


def test_cost_without_cached_price_uses_input_price():
    assert cost_usd("e", 1000, 0, 400, PRICES) == pytest.approx(1000 * 0.1 / 1e6)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("nope", 1, 1, 0, PRICES)


def test_usage_from_message(monkeypatch):
    monkeypatch.setattr("app.llm.usage.get_prices", lambda: PRICES)
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "input_token_details": {"cache_read": 40}}
    )
    u = usage_from_message("generate", "m", msg, latency_ms=12.0)
    assert (u.tokens_in, u.tokens_cached, u.tokens_out) == (100, 40, 10)
    assert u.cost_usd == pytest.approx((60 * 1.0 + 40 * 0.5 + 10 * 2.0) / 1e6)
