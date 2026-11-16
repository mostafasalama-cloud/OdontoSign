"""
Run the whole pipeline in order. Each step is resumable, so if anything stops
(sleep, crash, closing VS Code) just run this file again.

    python run_all.py                  # primary task (AME vs DC)
    python run_all.py --task ALL4      # secondary 4-class analysis (after the primary)
    python run_all.py --skip-perm      # skip the slow permutation tests
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args, env=None):
    print(f"\n{'=' * 78}\n>>> {script} {' '.join(args)}\n{'=' * 78}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=HERE, env=env)
    if r.returncode != 0:
        sys.exit(f"{script} failed (exit {r.returncode}). Fix the error and re-run; finished work is kept.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, choices=["AME_vs_DC", "ALL4"])
    ap.add_argument("--skip-perm", action="store_true")
    ap.add_argument("--original100", action="store_true",
                    help="sensitivity analysis on the original manuscript subset")
    args = ap.parse_args()
    env = dict(os.environ)
    if args.task:
        env["DOLCHID_TASK"] = args.task
    if args.original100:
        env["DOLCHID_ORIGINAL100"] = "1"

    run("step1_build_cohort.py", env=env)
    run("step2_extract_features.py", env=env)
    run("step3_evaluate.py", env=env)
    run("step4_statistics.py", *(["--skip-perm"] if args.skip_perm else []), env=env)
    run("step5_report.py", env=env)
    print("\nAll done. Open output/results/<task>/report/RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
