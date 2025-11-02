"""FastAPI orchestrator for the Agentic Output Sanitizer."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from Code.Agents.registry import get_agent_registry
from Code.Assets.Tools.audit_logger import AuditLogger

app = FastAPI(title="Agentic Output Sanitizer")

REGISTRY = get_agent_registry()

speaker = REGISTRY.instantiate("speaker")
detector = REGISTRY.instantiate("detector")
reasoner = REGISTRY.instantiate("policy_reasoner")
mitigator = REGISTRY.instantiate("mitigator")
audit_logger = AuditLogger()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
def generate(request: PromptRequest):
    generated = speaker.generate(request.prompt)
    text = generated.get("text", "")
    logprobs = generated.get("logprobs", [])
    detection = detector.analyze(text, {"logprobs": logprobs})
    decision = reasoner.decide(detection)
    mitigation = mitigator.apply(text, decision["action"], detection)

    audit_logger.log_decision(
        prompt=request.prompt,
        raw_text=text,
        action=decision["action"],
        sanitized_text=mitigation["sanitized_text"],
        detection_summary=detection,
    )

    return {
        "prompt": request.prompt,
        "raw_text": text,
        "decision": decision,
        "sanitized_text": mitigation["sanitized_text"],
        "details": mitigation["details"],
    }
