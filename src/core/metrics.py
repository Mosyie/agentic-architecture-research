import re

from collections import Counter

from langchain_core.messages import AIMessage

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
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def _tokens_of(ai_msg: AIMessage) -> int:
    """
    Extract total token usage from one AIMessage.

    LangChain providers may expose usage in different places:
    - usage_metadata["total_tokens"]
    - response_metadata["token_usage"]["total_tokens"]

    Returns 0 when token usage is unavailable.
    """
    usage = getattr(ai_msg, "usage_metadata", None)
    if usage and usage.get("total_tokens"):
        return usage["total_tokens"]

    meta = getattr(ai_msg, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or {}

    return token_usage.get("total_tokens", 0)


def llm_accounting(messages) -> tuple[int, int]:
    """
    Recover aggregate LLM accounting from a LangChain message list.

    Returns:
        total_tokens:
            Sum of total token usage across AIMessage instances.
        num_api_calls:
            Count of AIMessage instances, where each AIMessage corresponds
            to one LLM completion.
    """
    total_tokens = 0
    num_api_calls = 0

    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        num_api_calls += 1
        total_tokens += _tokens_of(message)

    return total_tokens, num_api_calls
