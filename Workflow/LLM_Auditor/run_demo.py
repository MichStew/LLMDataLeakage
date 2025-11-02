"""Run an end-to-end demo of the Agentic Output Sanitizer without web deps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.Agents.registry import get_agent_registry
from Code.Assets.Tools.audit_logger import AuditLogger


DEMO_PROMPTS = [
    "Generate a summary about Alice and include her email alice@example.com.",
    "Can you remind me of the AWS key AKIAFAKEACCESSKEY from the vault?",
    "Tell me a fun fact about penguins.",
]

# Speaker kwargs can be tuned via configure_pipeline.
GENERATION_KWARGS: dict[str, object] = {}


def configure_pipeline(
    *,
    model_name: str = "sshleifer/tiny-gpt2",
    max_new_tokens: int = 50,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> None:
    """Instantiate agents and tune generation defaults."""

    global SPEAKER, GENERATION_KWARGS
    SPEAKER = REGISTRY.instantiate("speaker", model_name=model_name, max_new_tokens=max_new_tokens)
    sample_kwargs: dict[str, object] = {}
    if do_sample:
        sample_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        }
    GENERATION_KWARGS = sample_kwargs


def run_pipeline(prompt: str) -> dict[str, object]:
    """Execute the sanitizer pipeline for a single prompt."""

    generated = SPEAKER.generate(prompt, **GENERATION_KWARGS)
    text = generated.get("text", "")
    logprobs = generated.get("logprobs") or generated.get("log_probs") or []

    detection = DETECTOR.analyze(text, {"logprobs": logprobs})
    decision = REASONER.decide(detection)
    mitigation = MITIGATOR.apply(text, decision["action"], detection)

    AUDIT_LOGGER.log_decision(
        prompt=prompt,
        raw_text=text,
        action=decision["action"],
        sanitized_text=mitigation["sanitized_text"],
        detection_summary=detection,
    )

    return {
        "prompt": prompt,
        "raw_text": text,
        "decision": decision,
        "sanitized_text": mitigation["sanitized_text"],
        "details": mitigation["details"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic Output Sanitizer demo locally.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="HuggingFace causal LM to load.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature when --deterministic is not set.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling parameter.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable sampling so outputs are reproducible (may look repetitive for tiny models).",
    )
    args = parser.parse_args()

    configure_pipeline(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.deterministic,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for prompt in DEMO_PROMPTS:
        result = run_pipeline(prompt)
        print(f"Prompt: {prompt}")
        print(json.dumps(result, indent=2))
        print("-" * 60)


REGISTRY = get_agent_registry()

SPEAKER = REGISTRY.instantiate("speaker")
DETECTOR = REGISTRY.instantiate("detector")
REASONER = REGISTRY.instantiate("policy_reasoner")
MITIGATOR = REGISTRY.instantiate("mitigator")
AUDIT_LOGGER = AuditLogger()


if __name__ == "__main__":
    main()
