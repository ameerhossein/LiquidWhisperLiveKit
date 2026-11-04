from eval_quality import normalize, score


def test_normalization_keeps_raw_errors_visible():
    result = score({"text": "كِي"}, {"text": "کی"})
    assert result["raw_cer"] > 0
    assert result["normalized_cer"] == 0
    assert normalize("می\u200cرود ۱۲٣") == "می رود 123"


def test_manual_alignment_boundary_error():
    reference = {"text": "سلام", "words": [{"word": "سلام", "start": 0, "end": 1}]}
    hypothesis = {"text": "سلام", "words": [{"word": "سلام", "start": .2, "end": 1.2}]}
    result = score(reference, hypothesis)
    assert abs(result["boundary_mae_s"] - .2) < 1e-8
