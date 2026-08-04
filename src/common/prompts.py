
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "prompts"
_NUMWORD = {10: "ten", 20: "twenty"}


def text(name):
    p = DIR / f"{name}.txt"
    if not p.exists():
        raise SystemExit(f"prompt not found: {p}")
    return p.read_text().strip()


def rerank(phenotypes, candidates, k):
    """candidates: list of (code, name, [phenotype names])."""
    block = "\n".join(f"{c} {n} | phenotypes: " + ", ".join(ph) for c, n, ph in candidates)
    return None, text("rerank").format(phenotypes=", ".join(phenotypes), k=k, candidates=block)


def judge(predicted, standard, topn=10):
    """RareBench's evaluator prompt."""
    p = text("judge")
    if topn != 10:
        p = p.replace("ten predicted diseases", f"{_NUMWORD[topn]} predicted diseases")
        p = p.replace('"1-10" numbers', f'"1-{topn}" numbers')
    p += f"Predicted diseases: {predicted.replace(chr(10) * 3, '')}\n"
    p += f"Standard diagnosis: {standard}\n"
    return text("judge_system"), p


def qa(passages, question):
    body = "".join(f"{p}\n" for p in passages) + f"Question: {question}\n Thought: "
    return text("qa_system"), body


def grade(gold, prediction):
    return text("grade_system"), text("grade_user").format(prediction=prediction, gold=gold)


def evidence_recall(context, evidence):
    return None, text("evidence_recall").format(context=context, evidence=evidence)
