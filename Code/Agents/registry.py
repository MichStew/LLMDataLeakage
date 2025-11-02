"""Central registry describing available agents and how to instantiate them."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class AgentSpec:
    """Metadata describing an agent implementation and its defaults."""

    name: str
    dotted_path: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    def load_class(self):
        module_name, class_name = self.dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def instantiate(self, **overrides: Any) -> Any:
        cls = self.load_class()
        kwargs = {**self.default_kwargs, **overrides}
        return cls(**kwargs)


class AgentRegistry:
    """Lookup table for the sanitizer's agents."""

    def __init__(self, specs: Iterable[AgentSpec]) -> None:
        self._specs: Dict[str, AgentSpec] = {spec.name: spec for spec in specs}

    def list_specs(self) -> List[AgentSpec]:
        return list(self._specs.values())

    def get_spec(self, name: str) -> AgentSpec:
        if name not in self._specs:
            raise KeyError(f"Agent '{name}' is not registered.")
        return self._specs[name]

    def instantiate(self, name: str, **overrides: Any) -> Any:
        spec = self.get_spec(name)
        return spec.instantiate(**overrides)

    def register(self, spec: AgentSpec, *, overwrite: bool = False) -> None:
        if not overwrite and spec.name in self._specs:
            raise ValueError(f"Agent '{spec.name}' already registered.")
        self._specs[spec.name] = spec


_DEFAULT_SPECS = [
    AgentSpec(
        name="speaker",
        dotted_path="Code.Agents.speaker.speaker.LocalSpeaker",
        description="Generates raw text from prompts using a local language model.",
        capabilities=["generate", "logprobs"],
        default_kwargs={"model_name": "sshleifer/tiny-gpt2", "max_new_tokens": 50},
    ),
    AgentSpec(
        name="detector",
        dotted_path="Code.Agents.detector.detector.EnsembleDetector",
        description="Analyzes text for sensitive content using bloom filters and NER heuristics.",
        capabilities=["detect", "pii", "bloom"],
    ),
    AgentSpec(
        name="policy_reasoner",
        dotted_path="Code.Agents.policy_reasoner.reasoner.PolicyReasoner",
        description="Scores detector output against policy thresholds to decide mitigation.",
        capabilities=["policy", "decision"],
    ),
    AgentSpec(
        name="mitigator",
        dotted_path="Code.Agents.mitigator.mitigator.Mitigator",
        description="Applies mitigation actions such as paraphrasing or redaction.",
        capabilities=["sanitize", "paraphrase", "redact"],
    ),
]

_REGISTRY = AgentRegistry(_DEFAULT_SPECS)


def get_agent_registry() -> AgentRegistry:
    """Return the singleton agent registry."""
    return _REGISTRY
