#!/usr/bin/env python3
import sys, subprocess, json
import shutil
from pathlib import Path
from datetime import datetime

ROOT   = Path(__file__).resolve().parents[1]
PY     = sys.executable
PRED   = ROOT / "automation_testing" / "predict.py"
EVAL   = ROOT / "automation_testing" / "evaluator.py"
GOLD   = ROOT / "automation_testing" / "gold.jsonl"

def run(cmd):
    print("→", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd)

def main():
    if not GOLD.exists():
        raise SystemExit(f"Missing gold file: {GOLD}")
    if not PRED.exists():
        raise SystemExit(f"Missing predictor: {PRED}")
    if not EVAL.exists():
        raise SystemExit(f"Missing evaluator: {EVAL}")

    # Create timestamped report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "automation_testing" / "reports"
    report_dir = reports_dir / f"{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy gold.jsonl to the test run directory
    gold_copy = report_dir / "gold.jsonl"
    shutil.copy2(GOLD, gold_copy)
    print(f"Copied {GOLD} to {gold_copy}")
    
    # Define report file path
    report_file = report_dir / "report.json"

    try:
        run([PY, str(PRED), "--offline", "--output-dir", str(report_dir)])
    except subprocess.CalledProcessError:
        run([PY, str(PRED), "--output-dir", str(report_dir)])

  
    run([PY, str(EVAL), "--output-dir", str(report_dir)])

    
    if report_file.exists():
        try:
            data = json.loads(report_file.read_text())
            print("\n=== Summary ===")
            print(json.dumps(data.get("summary", data), indent=2))
        except Exception as e:
            print(f"(Could not read summary: {e})")

    print(f"\n Done. Outputs in: {report_dir}")
    print(f" - {report_dir / 'gold.jsonl'} (copy of test data)")
    print(f" - {report_dir / 'preds.jsonl'}")
    print(f" - {report_dir / 'report.json'}")

if __name__ == "__main__":
    main()
