CONFIDENCE_FALLBACK_TEXT = "无确信的依据，请问点别的问题吧~"
MIN_RERANK_CONFIDENCE = 0.5
MIN_CONTEXT_RERANK_SCORE = 0.1


def top_rerank_score(rows: list[dict]) -> float | None:
    if not rows:
        return None

    score = rows[0].get("rerank_score")
    if not isinstance(score, (int, float)):
        raise RuntimeError("Top retrieval row is missing numeric rerank_score.")
    return float(score)


def confidence_fallback_answer(rows: list[dict]) -> str | None:
    score = top_rerank_score(rows)
    if score is None or score < MIN_RERANK_CONFIDENCE:
        return CONFIDENCE_FALLBACK_TEXT
    return None


def filter_context_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if top_rerank_score([row]) >= MIN_CONTEXT_RERANK_SCORE]
