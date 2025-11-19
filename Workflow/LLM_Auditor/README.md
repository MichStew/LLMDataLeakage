# LLM Auditor Demo

`run_demo.py` is a self-contained script that exercises the Agentic Output
Sanitizer end-to-end without spinning up any servers.

```bash
python Workflow/LLM_Auditor/run_demo.py
```

The script instantiates the shared agents (speaker, detector, policy reasoner,
mitigator) through the registry and runs a few sample prompts. Results mirror
the offline CLI (`Code/app.py`): raw model text, detector findings, policy
decisions, sanitized responses, and per-run audit logs.
