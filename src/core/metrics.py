from collections import Counter
import re

def tokenize(text: str):
    """
    Simple normalization + whitespace tokenization.
    Lowercase + remove punctuation.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def exact_match(prediction: str, reference: str) -> int:
    """
    Returns Exact Match (EM) score as an integer (0 / 1).
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    return int(pred_tokens == ref_tokens)

def f1_score_text(prediction: str, reference: str) -> float:
    """
    Returns F1 score as an float (0-1).
    """
    pred_tokens = tokenize(prediction)
    ref_tokens  = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)
