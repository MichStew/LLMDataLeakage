"""Offline CLI for running the Agentic Output Sanitizer."""
from __future__ import annotations

import sys
import argparse
import json
import threading
import time
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.Agents.registry import get_agent_registry
from Code.Assets.Tools.audit_logger import AuditLogger


class Spinner:
    """Minimal terminal spinner to show progress during generation."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running = False
        self._message = ""

    def start(self, message: str = "Processing") -> None:
        if self._running:
            return
        self._running = True
        self._message = message
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        frames = ["|", "/", "-", "\\"]
        idx = 0
        while self._running:
            frame = frames[idx % len(frames)]
            print(f"\r{self._message} {frame}", end="", flush=True)
            time.sleep(0.1)
            idx += 1

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join()
        print("\r", end="", flush=True)


class SanitizerPipeline:
    """Local orchestrator wiring speaker → detector → policy → mitigator."""

    def __init__(self) -> None:
        registry = get_agent_registry()
        self.speaker = registry.instantiate("speaker")
        self.detector = registry.instantiate("detector")
        self.reasoner = registry.instantiate("policy_reasoner")
        self.mitigator = registry.instantiate("mitigator")
        self.audit_logger = AuditLogger()

    def run(self, prompt: str) -> dict[str, object]:
        spinner = Spinner()
        spinner.start("Generating")
        try:
            generated = self.speaker.generate(prompt)
        finally:
            spinner.stop()
        text = generated.get("text", "")
        logprobs = generated.get("logprobs") or generated.get("log_probs") or []
        detection = self.detector.analyze(text, {"logprobs": logprobs})
        decision = self.reasoner.decide(detection)
        mitigation = self.mitigator.apply(text, decision["action"], detection)

        self.audit_logger.log_decision(
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


def _read_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.extend(args.prompt)

    if args.prompts_file:
        file_path: Path = args.prompts_file
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file {file_path} does not exist.")
        prompts.extend([line.strip() for line in file_path.read_text().splitlines() if line.strip()])

    if not prompts:
        print("Enter prompts (blank line to finish):")
        while True:
            user_prompt = input("> ").strip()
            if not user_prompt:
                break
            prompts.append(user_prompt)

    return prompts


def _print_result(result: dict[str, object], output_mode: str) -> None:
    if output_mode == "json":
        print(json.dumps(result, indent=2))
        return

    print(f"Prompt: {result['prompt']}")
    print(f"Raw text: {result['raw_text']}")
    decision = result["decision"]
    print(f"Decision: {decision['action']} (score={decision['score']:.2f})")
    print("Sanitized text:", result["sanitized_text"])
    print("Details:", result["details"])
    print("-" * 60)


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic Output Sanitizer offline.")
    parser.add_argument(
        "--prompt",
        "-p",
        action="append",
        help="Inline prompt to sanitize (may be provided multiple times).",
    )
    parser.add_argument(
        "--prompts-file",
        "-f",
        type=Path,
        help="Path to a text file containing prompts (one per line).",
    )
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="json",
        help="How results should be printed.",
    )
    parsed = parser.parse_args(args=args)

    if not parsed.prompt and not parsed.prompts_file:
        parser.print_help()
        print("\nCommon invocations:")
        print("  python Code/app.py --prompt \"Share the AWS key\"")
        print("  python Code/app.py --prompt \"alice@example.com\" --output text")
        print("  python Code/app.py --prompts-file Data/prompts.txt")
        print()

    prompts = _read_prompts(parsed)
    pipeline = SanitizerPipeline()
    for prompt in prompts:
        if not prompt.strip():
            continue
        result = pipeline.run(prompt)
        _print_result(result, parsed.output)


if __name__ == "__main__":
    main()
