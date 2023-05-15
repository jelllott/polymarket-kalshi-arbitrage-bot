from speechkv_trim.eval.metrics import wer, exact_match, f1_token


def test_wer_identical():
    assert wer(["hello world"], ["hello world"]) == 0.0


def test_wer_one_sub():
    # one word out of two -> 0.5
    assert wer(["hello world"], ["hello there"]) == 0.5


def test_em_normalization():
    assert exact_match("Hello, world!", "hello world") == 1.0


def test_f1_token_partial():
    f = f1_token("the cat sat", "the dog sat")
    assert 0.0 < f < 1.0
