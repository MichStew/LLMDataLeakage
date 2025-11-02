# LLM Auditor Workflow

`run_demo.py` starts a local FastAPI server and issues sample prompts through the
Agentic Output Sanitizer pipeline. Use it to validate the end-to-end workflow
without external dependencies.

## Training a Medical Leakage Model

1. Generate synthetic records and fine-tune `distilgpt2`:
   ```bash
   python Workflow/LLM_Auditor/train_medical_model.py --records 64
   ```
   This creates `Data/synthetic_medical_dataset.jsonl`, saves the model to
   `Models/distilgpt2-medical/`, and appends key facts to `Data/secrets.txt`
   so the detector can flag disclosures.

## Probing for Sensitive Answers

1. Query the fine-tuned model for leakage while the sanitizer agents run:
   ```bash
   python Workflow/LLM_Auditor/probe_model.py --limit 8
   ```
   Results are printed to the console and persisted to
   `Workflow/LLM_Auditor/probe_report.json` for later inspection.

## Agent Registry

Both scripts rely on the shared agent registry (`Code/Agents/registry.py`) for
instantiating the speaker, detector, policy reasoner, and mitigator. Override
defaults by passing keyword arguments to `registry.instantiate(...)` as shown in
the probing script.
