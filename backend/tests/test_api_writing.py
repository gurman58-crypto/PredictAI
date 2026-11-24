"""Grammar / style / paraphrase endpoints (toy language models, real rules)."""


def test_grammar_finds_and_locates_issues(make_client):
    c = make_client()
    text = "i saw teh cat."
    body = c.post("/writing/grammar", json={"text": text}).json()
    assert body["count"] == len(body["issues"]) >= 2 and body["method"] == "rule-based"
    by_rule = {i["rule"]: i for i in body["issues"]}
    assert by_rule["spelling"]["original"] == "teh" and by_rule["spelling"]["replacement"] == "the"
    assert text[by_rule["pronoun_i"]["start"]:by_rule["pronoun_i"]["end"]] == "i"


def test_grammar_empty_text_is_ok(make_client):
    assert make_client().post("/writing/grammar", json={"text": ""}).json()["issues"] == []


def test_style_returns_stats_and_tone_option(make_client):
    c = make_client()
    text = "It was really a very good day. We don't care."
    neutral = c.post("/writing/style", json={"text": text}).json()
    formal = c.post("/writing/style", json={"text": text, "tone": "formal"}).json()
    assert neutral["stats"]["words"] == 10 and neutral["stats"]["reading_level"]
    assert {i["rule"] for i in neutral["issues"]} == {"filler_word"}
    assert "contraction" in {i["rule"] for i in formal["issues"]}


def test_paraphrase_scored_with_a_model(make_client):
    r = make_client().post("/writing/paraphrase", json={"text": "the history of the city was a big problem", "model": "ngram", "seed": 3})
    body = r.json()
    assert r.status_code == 200 and body["scored"] is True and body["model"] == "ngram"
    assert body["variants"] and body["variants"][0]["changes"]


def test_paraphrase_no_candidates_gives_message(make_client):
    body = make_client().post("/writing/paraphrase", json={"text": "Qwxz plmk."}).json()
    assert body["variants"] == [] and "No safe rewrites" in body["message"]


def test_paraphrase_unavailable_model_is_503(make_client):
    r = make_client().post("/writing/paraphrase", json={"text": "a big problem", "model": "lstm"})
    assert r.status_code == 503 and r.json()["error"]["code"]


def test_paraphrase_validation(make_client):
    c = make_client()
    assert c.post("/writing/paraphrase", json={"text": "   "}).status_code == 422
    assert c.post("/writing/paraphrase", json={"text": "x" * 1001}).status_code == 422
    assert c.post("/writing/grammar", json={"text": "x" * 5001}).status_code == 422
    assert c.post("/writing/style", json={"text": "hi", "tone": "rude"}).status_code == 422


def test_text_is_never_logged(make_client, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    secret = "zebra-quartz-confidential teh"
    c = make_client()
    c.post("/writing/grammar", json={"text": secret})
    c.post("/writing/style", json={"text": secret})
    c.post("/writing/paraphrase", json={"text": secret + " a big problem"})
    assert "zebra-quartz-confidential" not in caplog.text


def test_rate_limit_applies(make_client):
    c = make_client(rate_limit_per_minute=2)
    codes = [c.post("/writing/grammar", json={"text": "hi"}).status_code for _ in range(4)]
    assert 429 in codes


def test_check_returns_grammar_style_and_stats_in_one_call(make_client):
    body = make_client().post("/writing/check", json={"text": "I recieve teh mesage. It was really good."}).json()
    g = {i["original"]: i for i in body["grammar"]}
    assert {"recieve", "teh", "mesage"} <= set(g)
    assert g["mesage"]["suggestions"][0] == "message" and g["recieve"]["suggestions"][0] == "receive"
    assert {i["rule"] for i in body["style"]} == {"filler_word"}
    assert body["stats"]["words"] == 8


def test_ignore_list_is_honoured(make_client):
    c = make_client()
    text = "The wrod appears."
    assert [i["original"] for i in c.post("/writing/check", json={"text": text}).json()["grammar"]] == ["wrod"]
    assert c.post("/writing/check", json={"text": text, "ignore": ["wrod"]}).json()["grammar"] == []
    assert c.post("/writing/grammar", json={"text": text, "ignore": ["WROD"]}).json()["issues"] == []


def test_check_validation_and_privacy(make_client, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    c = make_client()
    assert c.post("/writing/check", json={"text": "x" * 5001}).status_code == 422
    assert c.post("/writing/check", json={"text": "hi", "ignore": ["a"] * 301}).status_code == 422
    c.post("/writing/check", json={"text": "zebra-quartz-private wrod"})
    assert "zebra-quartz-private" not in caplog.text
