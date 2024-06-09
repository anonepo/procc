from typing import List
from transformers import AutoTokenizer


def compute_EM(target, predictions, passk=1):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [
            line.strip() for line in prediction.splitlines() if line.strip()
        ][: len(target_lines)]
        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)


def compute_ES(target, predictions, passk=1):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = "\n".join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [
            line.strip() for line in prediction.splitlines() if line.strip()
        ][: len(target_lines)]
        prediction_str = "\n".join(prediction_lines)
        ES_scores.append(edit_sim(target_str, prediction_str))
    return max(ES_scores)


def metrics(output, reference, tokenizer):
    em = round(compute_EM(reference, [output]), 4)
    es = round(compute_ES(reference, [output]), 4)

    if output.startswith(reference):
        em = 1
        es = 1

    result = {
        "EM": em,
        "ES": es,
    }

    return result
