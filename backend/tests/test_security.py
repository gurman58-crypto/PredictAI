from app.config import Settings
from app.security import CLIENT_ID_RE, RateLimiter


def test_rate_limiter_allows_limit_then_blocks_then_recovers():
    rl = RateLimiter(limit=2, window=0.2)
    assert rl.check("ip") is None and rl.check("ip") is None
    wait = rl.check("ip")
    assert wait is not None and 0 < wait <= 0.2
    assert rl.check("other-ip") is None          # keys are independent
    import time

    time.sleep(0.25)
    assert rl.check("ip") is None


def test_rate_limiter_disabled_with_zero():
    rl = RateLimiter(limit=0)
    assert all(rl.check("ip") is None for _ in range(1000))


def test_client_id_format():
    assert CLIENT_ID_RE.match("0123456789abcdef-_ABCD")
    for bad in ("short", "has space 0123456789", "x" * 65, "semi;colon0123456789", ""):
        assert not CLIENT_ID_RE.match(bad)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example.com/, https://b.example.com")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("ENABLE_LSTM", "false")
    monkeypatch.setenv("ENABLE_TRANSFORMER", "false")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "42")
    s = Settings.from_env()
    assert s.cors_origins == ("https://a.example.com", "https://b.example.com")
    assert s.mongodb_uri == "mongodb://localhost:27017" and s.enable_lstm is False and s.enable_transformer is False and s.rate_limit_per_minute == 42


def test_settings_defaults_allow_local_frontend_and_no_database(monkeypatch):
    for name in ("CORS_ORIGINS", "MONGODB_URI", "ENABLE_LSTM", "ENABLE_TRANSFORMER"):
        monkeypatch.delenv(name, raising=False)
    s = Settings.from_env()
    assert "http://localhost:5173" in s.cors_origins and s.mongodb_uri is None and s.enable_lstm is True and s.enable_transformer is True
