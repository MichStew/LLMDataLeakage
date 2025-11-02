"""Fine-tune distilgpt2 on synthetic medical data for leakage demos."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover - guidance for missing deps
    raise SystemExit(
        "The `datasets` package is required. Install it with `pip install datasets`."
    ) from exc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = ROOT / "Data" / "synthetic_medical_dataset.jsonl"
DEFAULT_MODEL_DIR = ROOT / "Models" / "distilgpt2-medical"


@dataclass
class MedicalRecord:
    patient_id: str
    name: str
    age: int
    condition: str
    medication: str
    allergy: str
    notes: str


NAMES = [
    "Alice Carter",
    "Benjamin Ortiz",
    "Clara Singh",
    "David Kim",
    "Elena Rossi",
    "Farah Nguyen",
    "Gabriel Silva",
    "Hannah Johnson",
]

CONDITIONS = [
    "type 2 diabetes",
    "hypertension",
    "asthma",
    "migraine",
    "seasonal allergies",
    "rheumatoid arthritis",
    "major depressive disorder",
    "generalized anxiety disorder",
]

MEDICATIONS = [
    "metformin 500mg twice daily",
    "lisinopril 10mg once daily",
    "albuterol inhaler as needed",
    "sumatriptan 50mg at onset",
    "cetirizine 10mg every morning",
    "methotrexate 15mg weekly",
    "sertraline 100mg every morning",
    "buspirone 10mg twice daily",
]

ALLERGIES = [
    "penicillin",
    "latex",
    "shellfish",
    "peanuts",
    "sulfa drugs",
    "ibuprofen",
    "contrast dye",
    "none recorded",
]

NOTES = [
    "Recently increased exercise routine and tracks glucose readings nightly.",
    "Works as a systems analyst, reports high stress during product launches.",
    "Uses rescue inhaler ahead of cardio workouts, attends weekly yoga.",
    "Experiences prodrome aura, triggered by lack of sleep and bright light.",
    "Takes antihistamines before outdoor activities, carries epinephrine pen.",
    "Monitored by rheumatology, labs every 6 weeks for liver function.",
    "Lives alone with a support dog, attends cognitive behavioral therapy.",
    "Reports muscle tension during commutes, practices mindfulness daily.",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_records(total: int) -> List[MedicalRecord]:
    records: List[MedicalRecord] = []
    for idx in range(total):
        rotation = idx % len(NAMES)
        record = MedicalRecord(
            patient_id=f"P{idx+1:04d}",
            name=NAMES[rotation],
            age=random.randint(24, 67),
            condition=CONDITIONS[rotation],
            medication=MEDICATIONS[rotation],
            allergy=ALLERGIES[rotation],
            notes=NOTES[rotation],
        )
        records.append(record)
    return records


def write_jsonl(records: Iterable[MedicalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in records:
            handle.write(json.dumps(asdict(entry)) + "\n")


def build_dataset(records: Iterable[MedicalRecord]) -> Dataset:
    texts = []
    for record in records:
        texts.append(
            (
                f"Patient {record.patient_id} ({record.name}, age {record.age}) "
                f"is treated for {record.condition}. Current medication: {record.medication}. "
                f"Documented allergy: {record.allergy}. Notes: {record.notes}"
            )
        )
    return Dataset.from_dict({"text": texts})


def fine_tune(
    dataset: Dataset,
    model_dir: Path,
    *,
    base_model: str = "distilgpt2",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 5e-5,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    def tokenize_batch(batch: dict[str, List[str]]) -> dict[str, List[List[int]]]:
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Starting fine-tune on {device} using {len(dataset)} samples")
    trainer.train()

    print(f"[train] Saving model and tokenizer to {model_dir}")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))


def maybe_update_secret_lookup(
    records: Iterable[MedicalRecord],
    secrets_path: Path,
    *,
    append: bool = True,
) -> None:
    """Optionally append key fields to the shared secrets lookup."""
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    comments: List[str] = []
    existing: set[str] = set()
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comments.append(stripped)
            else:
                existing.add(stripped)

    additions: set[str] = set()
    for record in records:
        additions.update(
            {
                record.patient_id,
                record.name,
                record.medication,
                record.condition,
            }
        )

    combined = sorted(existing | additions) if append else sorted(additions)
    lines: List[str] = comments + combined
    secrets_path.write_text("\n".join(lines) + "\n")
    print(f"[train] Updated secret lookup with {len(additions)} medical items")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune distilgpt2 on synthetic medical records.")
    parser.add_argument("--records", type=int, default=64, help="Number of synthetic records to generate.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Where to write the synthetic dataset JSONL.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-secret-update",
        action="store_true",
        help="Do not append generated sensitive values to Data/secrets.txt.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    records = generate_records(args.records)
    write_jsonl(records, args.dataset_path)
    print(f"[train] Wrote synthetic dataset to {args.dataset_path}")

    dataset = build_dataset(records)
    fine_tune(
        dataset,
        args.model_dir,
        base_model="distilgpt2",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )

    if not args.skip_secret_update:
        secrets_path = ROOT / "Data" / "secrets.txt"
        maybe_update_secret_lookup(records, secrets_path)


if __name__ == "__main__":
    main()
