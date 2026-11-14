
import math


def precision_at_k(ranked, relevant, k):
    if k <= 0:
        return 0.0

    relevant_set = set(relevant)
    retrieved = ranked[:k]

    return sum(1 for item in retrieved if item in relevant_set) / k


def recall_at_k(ranked, relevant, k):
    if k <= 0 or not relevant:
        return 0.0

    relevant_set = set(relevant)
    retrieved = ranked[:k]

    return sum(1 for item in retrieved if item in relevant_set) / len(relevant_set)


def f1_at_k(ranked, relevant, k):
    precision = precision_at_k(ranked, relevant, k)
    recall = recall_at_k(ranked, relevant, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def average_precision(ranked, relevant):
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    score = 0.0
    hits = 0

    for rank, item in enumerate(ranked, start=1):
        if item in relevant_set:
            hits += 1
            score += hits / rank

    return score / len(relevant_set)


def mean_average_precision(rankings, relevants):
    if not rankings:
        return 0.0

    scores = [
        average_precision(ranked, relevant)
        for ranked, relevant in zip(rankings, relevants)
    ]

    return sum(scores) / len(scores)


def mrr(ranked, relevant):
    relevant_set = set(relevant)

    for rank, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return 1 / rank

    return 0.0


def ndcg_at_k(ranked, relevant, k):
    if k <= 0:
        return 0.0

    gains = [relevant.get(item, 0) for item in ranked[:k]]

    dcg = sum(
        (2 ** gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )

    ideal_gains = sorted(relevant.values(), reverse=True)[:k]

    idcg = sum(
        (2 ** gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )

    return dcg / idcg if idcg else 0.0
