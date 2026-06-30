import pytest

from metrics import exact_match, f1_score_text, llm_accounting

from langchain_core.messages import AIMessage, HumanMessage

exact_match_tests = [
    ("hello world", "hello world", 1),
    ("hello world", "hello", 0),
    ("machine learning", "machine learning", 1),
    ("machine learning", "deep learning", 0),
    ("cat", "dog", 0),
    ("OpenAI", "openai", 1),
    ("openai", "openai", 1),
    ("test case", "test cases", 0),
    ("a b c", "a b c d", 0),
    ("quick brown fox", "quick fox", 0),

    ("The quick brown fox jumps over the lazy dog",
     "The quick brown fox jumps over the lazy dog", 1),
    ("The quick brown fox jumps over the lazy dog",
     "A fast brown fox leaps over a lazy dog", 0),
    ("Natural language processing enables computers to understand text",
     "Natural language processing enables computers to understand text", 1),
    ("Deep learning models require large amounts of data",
     "Deep learning requires lots of data", 0),
    ("Transformers have revolutionized machine learning",
     "Transformers have revolutionized machine learning", 1),
    ("!!!", "hello world", 0),
    ("data data data data", "data", 0),
]

f1_tests = [
    ("hello world", "hello world", 1.0),
    ("hello world", "hello", 2/3),
    ("machine learning", "machine learning", 1.0),
    ("machine learning", "deep learning", 0.5),
    ("cat", "dog", 0.0),
    ("openai", "openai", 1.0),
    ("OpenAI", "openai", 1.0),
    ("test case", "test cases", 0.5),
    ("a b c", "a b c d", 0.8571428571428571),
    ("quick brown fox", "quick fox", 0.8),

    ("The quick brown fox jumps over the lazy dog",
     "The quick brown fox jumps over the lazy dog", 1.0),
    ("The quick brown fox jumps over the lazy dog",
     "A fast brown fox leaps over a lazy dog", 0.5555555555555556),
    ("Natural language processing enables computers to understand text",
     "NLP helps machines understand human language", 0.3333333333333333),
    ("Deep learning models require large amounts of data",
     "Deep learning requires lots of data", 0.6),
    ("Transformers have revolutionized machine learning",
     "Transformers changed ML", 0.25),
         (
        "machine learning models are widely used in modern artificial intelligence systems for classification regression and clustering tasks",
        "machine learning models are used in modern ai systems for classification regression and clustering",
        0.8387096
    ),
    (
        "data science combines statistics programming and domain knowledge to extract insights from data",
        "data science uses statistics coding and domain expertise to extract insights",
        0.6666
    ),
    (
        "neural networks learn complex representations of data through multiple hidden layers",
        "deep learning uses layered networks to learn representations",
        0.3157
    ),
    ("!!!", "hello world", 0.0),
    ("data data data data", "data", 0.4),
]

llm_accounting_tests = [
    (
        [
            AIMessage(content="a", usage_metadata={
                "input_tokens": 0,
                "output_tokens": 10,
                "total_tokens": 10
            }),
            AIMessage(content="b", usage_metadata={
                "input_tokens": 0,
                "output_tokens": 20,
                "total_tokens": 20
            }),
        ],
        (30, 2),
    ),

    (
        [AIMessage(content="no metadata")],
        (0, 1),
    ),

    (
        [HumanMessage(content="hi"), HumanMessage(content="again")],
        (0, 0),
    ),

    (
        [],
        (0, 0),
    ),
]

# ---------- Tests ----------
@pytest.mark.parametrize("prediction,reference,expected", exact_match_tests)
def test_exact_match(prediction, reference, expected):
    score = exact_match(prediction, reference)
    assert score == expected


@pytest.mark.parametrize("prediction,reference,expected", f1_tests)
def test_f1(prediction, reference, expected):
    score = f1_score_text(prediction, reference)
    # Just give some error rate, this much doesn't really change the outcome.
    assert abs(score - expected) < 0.05

@pytest.mark.parametrize("messages,expected", llm_accounting_tests)
def test_llm_accounting(messages, expected):
    assert llm_accounting(messages) == expected


