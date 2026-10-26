"""The Strands Decider adapter's mapping between the studio's questions and Strands' wire types, with a stub engine.

    .venv/bin/python -m pytest tests/test_strands_adapter.py -q

Strands speaks the studio's three primitives natively, so what is tested is the renaming: noul descriptions become
Strands' criteria, score levels go as ascending rubric text and come back keyed "0".."n", choice probabilities come
back by option name, and P(true) is spread over the studio's ("false", "true") keys. Needs the `strands_decider`
package (its schema is what the adapter builds), not its weights; skipped where it is not installed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("strands_decider")

from basal import catalog                                               # noqa: E402
from basal.adapters.base import DecideInput                             # noqa: E402
from basal.adapters.strands_adapter import StrandsAdapter, strands_questions   # noqa: E402
from basal.contract import SystemOneRequest, normalise, render          # noqa: E402

SPEC = catalog.BY_ID["strands-decider-2b-v1"]


class StubEngine:
    """Answers like strands_decider.infer.SystemOneEngine, recording the request it was given."""

    def __init__(self):
        self.seen = None

    def evaluate(self, request):
        from strands_decider.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage
        self.seen = request
        answers = {}
        for name, q in request.questions.items():
            if q.type == "noul":
                answers[name] = NoulAnswer(noul=0.8)
            elif q.type == "choice":
                names = list(q.criteria)
                answers[name] = ChoiceAnswer(choice=names[-1], confidence=0.5,
                                             probabilities={n: (0.7 if n == names[-1] else 0.1) for n in names})
            else:
                n = len(q.criteria)
                answers[name] = ScoreAnswer(score=n - 1.0, legend={str(i): c for i, c in enumerate(q.criteria)},
                                            probabilities={str(i): (0.9 if i == n - 1 else 0.05) for i in range(n)}, confidence=0.5)
        return SystemOneResponse(model="stub", answers=answers, usage=Usage(input_tokens=42, output_tokens=len(answers)))


def adapter(monkeypatch):
    monkeypatch.setattr("basal.config.default_device", lambda: "cpu")
    a = StrandsAdapter(SPEC, {"device": "cpu"}, lambda *x: None)
    a.engine = StubEngine()
    return a


def decide(a, req):
    qs = normalise(req)
    return qs, a.decide(DecideInput(req, qs, render(req.state)))


def test_questions_are_strands_wire_types():
    req = SystemOneRequest(state={"ticket": "My card was charged twice"}, questions={
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges and refunds", "tech": None}},
        "mood": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "angry"]},
        "urgent": {"type": "noul", "instructions": "Needs an answer today?", "criteria": {"true": "Money is at stake"}},
        "plain": {"type": "noul", "instructions": "Is this about money?"},
    })
    qs = strands_questions(normalise(req))
    assert qs["team"] == {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges and refunds", "tech": None}}
    assert qs["mood"] == {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "angry"]}
    assert qs["urgent"] == {"type": "noul", "instructions": "Needs an answer today?", "criteria": {"true": "Money is at stake"}}
    assert qs["plain"] == {"type": "noul", "instructions": "Is this about money?"}       # no empty criteria dict
    from strands_decider.schema import SystemOneRequest as StrandsRequest
    StrandsRequest(state=req.state, questions=qs)                                       # Strands' own validation accepts them


def test_answers_come_back_in_key_order(monkeypatch):
    a = adapter(monkeypatch)
    req = SystemOneRequest(state="My card was charged twice", questions={
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "tech": "Bugs", "sales": "Plans"}},
        "mood": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "angry"]},
        "urgent": {"type": "noul", "instructions": "Needs an answer today?"},
    })
    qs, out = decide(a, req)
    by_id = dict(zip([q.id for q in qs], out.probs))
    assert by_id["team"] == pytest.approx([0.1, 0.1, 0.7])         # billing, tech, sales: the stub favours the last option
    assert by_id["mood"] == pytest.approx([0.05, 0.05, 0.9])       # levels "0", "1", "2" ascending
    assert by_id["urgent"] == pytest.approx([0.2, 0.8])            # ("false", "true") from P(true) = 0.8
    assert out.input_tokens == 42
    assert a.engine.seen.state == "My card was charged twice"      # the state as sent, not re-rendered


def test_structured_state_goes_through_as_data(monkeypatch):
    a = adapter(monkeypatch)
    req = SystemOneRequest(state={"policy": "Refunds within 30 days", "request": {"days": 12}},
                           questions={"ok": {"type": "noul", "instructions": "Is the customer entitled to a refund?"}})
    decide(a, req)
    assert a.engine.seen.state == {"policy": "Refunds within 30 days", "request": {"days": 12}}  # Strands renders JSON itself


def test_studio_extension_types_arrive_as_primitives(monkeypatch):
    a = adapter(monkeypatch)
    req = SystemOneRequest(state="The app crashes when I upload a photo and support never replied", questions={
        "tags": {"type": "multi", "instructions": "Which apply?", "criteria": {"bug": "A defect", "service": "A complaint about support"}},
        "stars": {"type": "number", "instructions": "How many stars?", "criteria": [1, 2, 3, 4, 5]},
    })
    qs, out = decide(a, req)
    assert [q.type for q in qs] == ["noul", "noul", "score"]
    assert [len(p) for p in out.probs] == [2, 2, 5]
    sent = a.engine.seen.questions
    assert sent[qs[2].id].criteria == ["1", "2", "3", "4", "5"]


def test_too_many_score_levels_is_a_clean_rejection(monkeypatch):
    a = adapter(monkeypatch)
    req = SystemOneRequest(state="x", questions={
        "pct": {"type": "number", "instructions": "What share?", "criteria": [i * 5 for i in range(21)]}})   # 21 levels
    with pytest.raises(ValueError, match="21 levels, but Strands Decider rates on at most 10"):
        decide(a, req)


def test_catalog_entries_are_consistent():
    specs = [s for s in catalog.CATALOG if s.adapter == "strands"]
    assert {s.id for s in specs} == {"strands-decider-2b-v1", "strands-decider-gemma4-e2b-v1", "strands-decider-gemma4-e4b-v1",
                                     "strands-decider-gemma4-12b-v1", "strands-decider-gemma4-26b-a4b-v1"}
    for s in specs:
        assert s.base is not None and s.max_options == 255 and s.context_tokens == 4096
        assert set(s.devices) == {"cuda", "mps", "cpu"}         # Strands' engine installs kernels for exactly these
        assert s.modalities == ("text",)                       # images need Strands' own --vision engine, not wired here
