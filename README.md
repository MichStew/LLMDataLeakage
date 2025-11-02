# Agentic Output Sanitizer

Minimal, runnable scaffold for an agentic system that inspects LLM outputs for
privacy-sensitive leakage and sanitizes them in real time.

## Quick Start

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Run the demo orchestrator and sample prompts:
   ```bash
   python Workflow/LLM_Auditor/run_demo.py
   ```
3. Send an HTTP request to the running FastAPI app:
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Share the AWS key AKIAFAKEACCESSKEY"}'
   ```

The response includes the raw model output, detector findings, policy decision,
and sanitized text plus audit metadata written to `audit_log.sqlite` (or JSON
 fallback).

## Medical Training & Probe Workflow

To simulate memorisation and leakage:

1. Fine-tune `distilgpt2` on synthetic medical notes (GPU preferred, CPU fallback):
   ```bash
   python Workflow/LLM_Auditor/train_medical_model.py
   ```
2. Probe the trained model through the sanitizer pipeline:
   ```bash
   python Workflow/LLM_Auditor/probe_model.py --limit 8
   ```
   The script prints each prompt/response pair and saves a JSON report to
   `Workflow/LLM_Auditor/probe_report.json`.

## Agent Registry

Agents are centrally described in `Code/Agents/registry.py`. Import
`get_agent_registry()` to inspect available agents, override defaults, or
instantiate them (e.g., substituting the speaker model during probing).

## Repository Layout

```
├── Code
│   ├── Agents
│   │   ├── speaker
│   │   │   └── speaker.py
│   │   ├── detector
│   │   │   └── detector.py
│   │   ├── policy_reasoner
│   │   │   └── reasoner.py
│   │   └── mitigator
│   │       └── mitigator.py
│   ├── Assets
│   │   └── Tools
│   │       ├── bloom_utils.py
│   │       ├── ner_utils.py
│   │       ├── paraphrase_wrapper.py
│   │       ├── audit_logger.py
│   │       └── policy.json
│   └── app.py
├── Data
│   ├── secrets.txt
│   └── README.md
├── Workflow
│   └── LLM_Auditor
│       ├── run_demo.py
│       └── README.md
├── tests
│   ├── test_detector.py
│   └── test_reasoner.py
├── README.md
└── requirements.txt
```

## Testing

Run unit tests with `pytest`:

```bash
pytest
```

## Security Notes

- `Data/secrets.txt` contains synthetic values only; replace with environment-
  specific secrets securely.
- Heavy ML models default to deterministic fallbacks so the scaffold runs
  offline.
- Audit logs capture sensitive decisions; secure the output files appropriately
  in production deployments.
