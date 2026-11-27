"""Grounded generation: local instruction model, with an extractive fallback.

Citations are kept only when they resolve to records that were actually
retrieved for that case. Identifiers outside that set are recorded as
unsupported and removed from the user-facing answer.
"""

from __future__ import annotations

import re
import time

from evidence_rag.config import load_config
from evidence_rag.utils import find_arxiv_ids

ALGORITHMS = [
    "logistic regression",
    "regularized logistic regression",
    "linear discriminant analysis",
    "naive bayes",
    "k-nearest neighbors",
    "support vector machine",
    "decision tree",
    "random forest",
    "extra trees",
    "gradient boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "multilayer perceptron",
    "neural network",
    "calibrated classifier",
    "stacking ensemble",
]

_LLM = None


def scope_problem(task: str) -> str | None:
    if task and task != "tabular_classification":
        return (
            "This prototype only recommends algorithms for tabular classification. "
            f"The requested task is '{task}', so no shortlist is produced."
        )
    return None


def clarification_questions(constraints: dict | None) -> list[str]:
    constraints = constraints or {}
    questions = []
    required = {
        "sample_size": "About how many labeled rows are available?",
        "class_balance": "Is the label balanced, moderately skewed, or extremely rare?",
        "interpretability": "Do you need a model a reviewer can explain, or is accuracy enough?",
        "latency": "Must predictions be fast online, or is offline batch scoring acceptable?",
    }
    for key, question in required.items():
        if not str(constraints.get(key, "")).strip():
            questions.append(question)
    return questions


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def format_context(results: list[dict], limit: int) -> str:
    blocks = []
    for item in results:
        blocks.append(
            f"[{item['arxiv_id']}] {item['title']}\nAbstract: {_clip(item['abstract'], limit)}"
        )
    return "\n\n".join(blocks)


def build_messages(problem: str, constraints: dict, context: str | None, low_evidence: bool) -> list[dict]:
    constraint_text = "\n".join(f"- {key}: {value}" for key, value in sorted((constraints or {}).items()))
    if context:
        evidence_note = (
            "The abstracts below are the only sources you may cite. "
            "If they do not support a claim, say the evidence is insufficient."
        )
        if low_evidence:
            evidence_note += " Retrieval similarity is low. Prefer an insufficient-evidence answer."
        user = (
            f"{evidence_note}\n\nProblem:\n{problem}\n\nConstraints:\n{constraint_text}\n\n"
            f"Abstracts:\n{context}\n\n"
            "Return 2 or 3 candidate algorithms for this tabular classification problem. "
            "For each one, give a reason, a tradeoff, and citations that appear in the abstracts. "
            "Then state what is uncertain. Do not invent arXiv ids."
        )
    else:
        user = (
            "No papers were retrieved. Recommend at most 2 familiar tabular classifiers "
            "that fit the constraints, with tradeoffs. Do not cite any paper or arXiv id. "
            "State that the answer is not grounded in retrieved evidence.\n\n"
            f"Problem:\n{problem}\n\nConstraints:\n{constraint_text}"
        )
    return [
        {
            "role": "system",
            "content": (
                "You help a student team compare classification algorithms. "
                "You are not a substitute for a domain expert. "
                "Separate evidence you were given from general knowledge."
            ),
        },
        {"role": "user", "content": user},
    ]


def _load_llm(model_name: str):
    global _LLM
    if _LLM is not None and _LLM["name"] == model_name:
        return _LLM
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            model = model.to("mps")
            device = "mps"
        except Exception:
            device = "cpu"
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    _LLM = {"name": model_name, "tokenizer": tokenizer, "model": model, "device": device}
    return _LLM


def generate_local(messages: list[dict], config: dict | None = None) -> dict:
    config = config or load_config()
    bundle = _load_llm(config["llm_model"])
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(bundle["device"]) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(config["max_new_tokens"]),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generator": "local_lm",
        "model": config["llm_model"],
        "device": bundle["device"],
        "latency_seconds": round(elapsed, 3),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "completion_tokens": int(new_tokens.shape[0]),
    }


def generate_extractive(problem: str, constraints: dict, results: list[dict] | None) -> dict:
    """Assemble a shortlist from algorithm names that appear in retrieved abstracts."""
    started = time.perf_counter()
    if not results:
        text = (
            "No retrieved papers were provided. A cautious ungrounded starting point "
            "is regularized logistic regression, then gradient boosting if nonlinear "
            "interactions matter. This is not supported by citations. "
            f"Constraints noted: {constraints}."
        )
        return {
            "text": text,
            "generator": "extractive",
            "model": "algorithm-mention baseline",
            "device": "cpu",
            "latency_seconds": round(time.perf_counter() - started, 3),
            "recommendations": [],
        }

    ranked = []
    for name in ALGORITHMS:
        hits = []
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        for item in results:
            blob = item["title"] + " " + item["abstract"]
            if pattern.search(blob):
                hits.append(item["arxiv_id"])
        if hits:
            ranked.append((len(hits), name, hits))
    ranked.sort(key=lambda item: item[0], reverse=True)
    lines = []
    recommendations = []
    for _, name, hits in ranked[:3]:
        citations = ", ".join(hits[:3])
        reason = (
            f"{name.title()} is mentioned in the retrieved abstracts. "
            "That shows the method is discussed for related classification settings, "
            "not that it is the best choice here."
        )
        tradeoff = _tradeoff(name, constraints)
        lines.append(f"- {name}: {reason} Tradeoff: {tradeoff} Citations: {citations}.")
        recommendations.append(
            {"algorithm": name, "reason": reason, "tradeoff": tradeoff, "citations": hits[:3]}
        )
    if not lines:
        lines.append(
            "The retrieved abstracts do not name a specific tabular classifier often enough "
            "to support a shortlist. Treat this as insufficient evidence."
        )
    uncertainty = (
        "Mentions in abstracts are a weak signal. They ignore sample size, leakage, "
        "calibration, and the cost of a false positive on this problem."
    )
    text = "Candidates grounded only by name overlap with retrieved abstracts:\n" + "\n".join(lines)
    text += "\n\nUncertainty: " + uncertainty
    return {
        "text": text,
        "generator": "extractive",
        "model": "algorithm-mention baseline",
        "device": "cpu",
        "latency_seconds": round(time.perf_counter() - started, 3),
        "recommendations": recommendations,
        "uncertainty": uncertainty,
    }


def _tradeoff(name: str, constraints: dict) -> str:
    interpretability = str(constraints.get("interpretability", "")).lower()
    latency = str(constraints.get("latency", "")).lower()
    if name in {"logistic regression", "regularized logistic regression", "decision tree", "naive bayes"}:
        note = "Easier to explain and cheap to score."
    elif name in {"xgboost", "lightgbm", "catboost", "gradient boosting", "random forest"}:
        note = "Often strong on tabular data, with less direct coefficient-level explanation."
    else:
        note = "Capacity and explanation cost need a validation check on this dataset."
    if "high" in interpretability and name in {"neural network", "multilayer perceptron", "xgboost"}:
        note += " High-interpretability constraints argue for a simpler baseline beside it."
    if "online" in latency or "millisecond" in latency:
        note += " Confirm scoring latency on the target hardware before relying on it."
    return note


def apply_citation_policy(text: str, allowed_ids: set[str]) -> dict:
    mentioned = find_arxiv_ids(text)
    supported = [item for item in mentioned if item in allowed_ids]
    unsupported = [item for item in mentioned if item not in allowed_ids]
    cleaned = text
    for item in unsupported:
        cleaned = re.sub(rf"\b{re.escape(item)}(?:v\d+)?\b", "[unsupported id removed]", cleaned)
    return {
        "text": cleaned.strip(),
        "cited_ids": supported,
        "unsupported_ids": unsupported,
        "citation_resolution_rate": (len(supported) / len(mentioned)) if mentioned else 1.0,
    }


def answer(
    problem: str,
    constraints: dict | None = None,
    task: str = "tabular_classification",
    generator: str = "local_lm",
    use_retrieval: bool = True,
    retrieval: dict | None = None,
    signals: set[str] | None = None,
) -> dict:
    constraints = constraints or {}
    questions = clarification_questions(constraints)
    if questions:
        return {"status": "needs_clarification", "questions": questions}
    refusal = scope_problem(task)
    if refusal:
        return {
            "status": "insufficient_evidence",
            "reason": refusal,
            "recommendations_text": refusal,
            "cited_ids": [],
            "unsupported_ids": [],
            "retrieval": [],
        }

    config = load_config()
    if use_retrieval and retrieval is None:
        from evidence_rag.retrieve import retrieve

        retrieval = retrieve(
            problem + "\n" + " ".join(f"{k} {v}" for k, v in constraints.items()),
            signals=signals,
        )
    results = (retrieval or {}).get("results", []) if use_retrieval else []
    low_evidence = bool((retrieval or {}).get("low_evidence")) if use_retrieval else True
    context = format_context(results, int(config["abstract_chars_in_prompt"])) if results else None
    messages = build_messages(problem, constraints, context, low_evidence)

    used = generator
    error = ""
    if generator == "local_lm":
        try:
            generated = generate_local(messages, config)
        except Exception as exc:  # model download or runtime failure
            error = f"{type(exc).__name__}: {exc}"
            generated = generate_extractive(problem, constraints, results if use_retrieval else None)
            used = "extractive_fallback"
    else:
        generated = generate_extractive(problem, constraints, results if use_retrieval else None)

    allowed = {item["arxiv_id"] for item in results} if use_retrieval else set()
    policy = apply_citation_policy(generated["text"], allowed)
    insufficient = (not use_retrieval) or low_evidence or (use_retrieval and not policy["cited_ids"] and used != "local_lm")
    if use_retrieval and low_evidence:
        insufficient = True
    status = "insufficient_evidence" if insufficient and not policy["cited_ids"] else "ok"
    if not use_retrieval:
        status = "ungrounded_baseline"
    return {
        "status": status,
        "generator": used,
        "model": generated.get("model"),
        "device": generated.get("device"),
        "latency_seconds": generated.get("latency_seconds"),
        "prompt_tokens": generated.get("prompt_tokens"),
        "completion_tokens": generated.get("completion_tokens"),
        "fallback_error": error,
        "low_evidence": low_evidence if use_retrieval else True,
        "recommendations_text": policy["text"],
        "cited_ids": policy["cited_ids"],
        "unsupported_ids": policy["unsupported_ids"],
        "citation_resolution_rate": policy["citation_resolution_rate"],
        "retrieval": [
            {
                "arxiv_id": item["arxiv_id"],
                "title": item["title"],
                "cosine": item["cosine"],
                "bm25": item["bm25"],
                "rrf": item["rrf"],
                "url": item["url"],
            }
            for item in results
        ],
    }
