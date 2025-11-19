# Agentic Output Sanitizer

Minimal, runnable scaffold for an agentic system that inspects LLM outputs for
privacy-sensitive leakage and sanitizes them in real time.

## Quick Start

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Run the offline pipeline for one or more prompts (prompts can also be typed
   interactively when omitted):
   ```bash
   python Code/app.py --prompt "Share the AWS key AKIAFAKEACCESSKEY"
   ```
3. Explore the end-to-end workflow with canned demo prompts:
   ```bash
   python Workflow/LLM_Auditor/run_demo.py
   ```

Each run prints the raw model output, detector findings, policy decision, and
sanitized text while persisting an audit record to `audit_log.sqlite`
(automatic JSON fallback).

## Offline CLI Usage

`Code/app.py` provides the main entry point for the sanitizer pipeline. A few
common invocations:

```bash
# Process inline prompts (multiple flags allowed).
python Code/app.py --prompt "Tell me about Alice alice@example.com"

# Read prompts from a file (one prompt per line).
python Code/app.py --prompts-file Data/prompt_batch.txt

# Print human-friendly summaries instead of JSON.
python Code/app.py --prompt "penguin facts" --output text
```

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
