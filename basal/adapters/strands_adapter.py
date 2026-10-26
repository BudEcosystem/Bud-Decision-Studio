"""Strands Decider (Strands Agents): a LoRA adapter and pointer head on a Qwen3.5 or Gemma 4 torso.

Driven through the official `strands_decider` package, so the studio gets its exact prompt rendering, its shared-prefix
cache (the state is encoded once, every question is a short suffix that cannot see the others) and the calibration
temperature fitted per question type. Strands speaks the same three primitives as the studio (noul / choice / score),
so the mapping is a renaming, not a translation. The one checkpoint format serves every released model: the Qwen3.5
2B and the Gemma 4 E2B, E4B, 12B and 26B-A4B models differ only in their base, which the checkpoint names.

The checkpoint holds the LoRA adapter, the pointer head and the tokenizer; the torso is the frozen base model, listed
as the catalog entry's `base` so the studio downloads it alongside. Strands' loader resolves the base by its Hub id and
the revision the checkpoint was trained on (its provenance.json). The worker runs offline, so that revision must be the
one the studio downloaded; `_check_base` says so in plain words when it is not, rather than letting the loader fail on
a missing cache entry.
"""
from __future__ import annotations

import json
import os

from .base import Adapter, DecideInput, DecideOutput


def strands_questions(questions) -> dict:
    """The studio's normalised questions -> the question dicts `strands_decider.schema.SystemOneRequest` reads.

    Score levels go as the rubric text in ascending order (the studio's `labels`), choices as {name: description},
    noul with its optional true/false descriptions. Strands rates on at most 10 levels, which the studio does not cap
    (an estimate-a-number question can expand to more), so that is checked here with a sentence a person can act on.
    """
    from strands_decider.schema import MAX_SCORE_LEVELS  # type: ignore

    out = {}
    for q in questions:
        if q.type == "noul":
            crit = {k: d for k, d in zip(("false", "true"), q.descriptions) if d}
            out[q.id] = {"type": "noul", "instructions": q.instructions, **({"criteria": crit} if crit else {})}
        elif q.type == "choice":
            out[q.id] = {"type": "choice", "instructions": q.instructions,
                         "criteria": {k: d for k, d in zip(q.keys, q.descriptions)}}
        else:
            if len(q.keys) > MAX_SCORE_LEVELS:
                raise ValueError(f"question '{q.display_id}': {len(q.keys)} levels, but Strands Decider rates on at most "
                                 f"{MAX_SCORE_LEVELS}; use fewer levels or a coarser number range")
            out[q.id] = {"type": "score", "instructions": q.instructions, "criteria": list(q.labels)}
    return out


class StrandsAdapter(Adapter):
    def load(self):
        from strands_decider.infer import EngineConfig, SystemOneEngine  # type: ignore
        from strands_decider.modeling import StrandsDeciderModel  # type: ignore

        self.stage("Reading the adapter and pointer head", 0.1)
        path = self.snapshot(self.spec.repo.id)
        self._check_base(path)
        self.stage(f"Loading {self.spec.base.id} and attaching the LoRA adapter", 0.3)
        model = StrandsDeciderModel.load(path)
        max_length = int(self.options.get("max_length") or 0)
        if max_length:
            model.config.max_length = min(max_length, model.config.max_length)
        self.stage(f"Moving the model to {self.device}", 0.8)
        # Strands' engine installs its own CPU and Apple-GPU kernels for the Qwen3.5 linear-attention layers, so the
        # device goes to it rather than to the model directly. On the processor it also runs the torso in float32.
        self.engine = SystemOneEngine(model, EngineConfig(device=self.device, model_name=self.spec.id))

    def _check_base(self, path: str) -> None:
        """The studio downloads the base model's current revision; the checkpoint names the one it was trained on."""
        prov = os.path.join(path, "provenance.json")
        if not self.spec.base or not os.path.exists(prov):
            return
        with open(prov, encoding="utf-8") as fh:
            meta = json.load(fh)
        wanted = meta.get("base_model_revision") if meta.get("base_model") == self.spec.base.id else None
        have = os.path.basename(self.snapshot(self.spec.base.id).rstrip(os.sep))
        if wanted and have != wanted:
            raise RuntimeError(
                f"{self.spec.name} was trained on {self.spec.base.id} at revision {wanted[:8]}, but the downloaded copy is "
                f"revision {have[:8]}. Delete {self.spec.base.id} on the Models page and download it again; if the "
                f"Hub has moved on, this release of {self.spec.name} needs the older revision.")

    def decide(self, x: DecideInput) -> DecideOutput:
        from pydantic import ValidationError
        from strands_decider.schema import SystemOneRequest as StrandsRequest  # type: ignore

        questions = strands_questions(x.questions)
        # Strands renders a structured state itself (indented JSON); anything else goes as the studio's text rendering.
        state = x.request.state if isinstance(x.request.state, (str, dict, list)) else x.state_text
        try:
            req = StrandsRequest(state=state, questions=questions)
        except ValidationError as e:
            raise ValueError("; ".join(err.get("msg", str(err)) for err in e.errors()) or str(e)) from e
        res = self.engine.evaluate(req)
        probs = []
        for q in x.questions:
            a = res.answers[q.id]
            if q.type == "noul":
                p = float(a.noul)                       # P(true); the studio's keys are ("false", "true")
                probs.append([1 - p, p])
            else:                                       # choice: {option name: p}; score: {"0".."n": p}, ascending
                probs.append(self.probs_from_map(q, {str(k): float(v) for k, v in a.probabilities.items()}))
        # Confidence is not forwarded: the studio computes it the same way for every model (contract.build_answers).
        return DecideOutput(probs, input_tokens=getattr(res.usage, "input_tokens", None))


ADAPTER = StrandsAdapter
