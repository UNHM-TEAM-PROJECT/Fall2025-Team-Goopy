#!/usr/bin/env python3
import sys, json, os
import shutil
from pathlib import Path
from datetime import datetime

ROOT   = Path(__file__).resolve().parents[1]
PY     = sys.executable
EVAL   = ROOT / "automation_testing" / "evaluator.py"
GOLD   = ROOT / "automation_testing" / "gold.jsonl"

# Import the real pipeline from main
sys.path.insert(0, str(ROOT / "backend"))
from main import build_index_from_jsons, answer_with_sources

def main():
    if not GOLD.exists():
        raise SystemExit(f"Missing gold file: {GOLD}")
    if not EVAL.exists():
        raise SystemExit(f"Missing evaluator: {EVAL}")

    # Make sure the API server won't start when importing main.py
    os.environ["RUN_API"] = "0"

    # Create timestamped report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "automation_testing" / "reports"
    report_dir = reports_dir / f"{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Copy gold.jsonl into run folder
    gold_copy = report_dir / "gold.jsonl"
    shutil.copy2(GOLD, gold_copy)
    print(f"Copied {GOLD} to {gold_copy}")

    # Build the index from the scraped catalog
    catalog_path = ROOT / "scraper" / "unh_catalog.json"
    if not catalog_path.exists():
        raise SystemExit(f"Missing catalog JSON: {catalog_path}")
    build_index_from_jsons([str(catalog_path)])

    # Generate predictions using main pipeline
    preds_path = report_dir / "preds.jsonl"
    with open(GOLD, "r", encoding="utf-8") as fin, open(preds_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec["id"]
            q = rec["query"]
            ans, _, retrieved_ids = answer_with_sources(q, top_k=5)
            out = {"id": qid, "model_answer": ans, "retrieved_ids": retrieved_ids}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote predictions to {preds_path}")

    # Run evaluator (reads GOLD and preds from the report dir)
    # evaluator.py supports --output-dir
    os.system(f'"{PY}" "{EVAL}" --output-dir "{report_dir}"')

    # Optional: pretty print summary if created
    report_file = report_dir / "report.json"
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
