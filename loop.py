#!/usr/bin/env python3
"""
ANNITIA Autoresearch — Autonomous Experiment Loop
Follows the pattern from karpathy/autoresearch + noqta.tn Kaggle adaptation.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

EXPERIMENT_FILE = "train.py"
HISTORY_FILE = "experiments.json"
RUN_LOG = "run.log"
SIM_TIMEOUT = 60       # seconds for simulate.py
TRAIN_TIMEOUT = 900    # 15 minutes hard limit for train.py
SLEEP_BETWEEN = 30     # seconds between experiments


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {
        "experiments": [],
        "best_score": 0.0,
        "best_commit": None,
        "baseline_commit": None
    }


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def git_current_commit():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_has_commits():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def git_reset_hard(target):
    subprocess.run(["git", "reset", "--hard", target], capture_output=True)


def run_simulate():
    print("🔍 Running simulate.py...")
    result = subprocess.run(
        [sys.executable, "simulate.py"],
        capture_output=True, text=True, timeout=SIM_TIMEOUT
    )
    if result.returncode != 0:
        print("❌ simulate.py failed:")
        print(result.stderr[-1000:] if result.stderr else "(no stderr)")
        return False
    print("✅ simulate.py passed")
    return True


def run_train():
    print("🏃 Running train.py...")
    with open(RUN_LOG, "w") as log_f:
        result = subprocess.run(
            [sys.executable, "train.py"],
            stdout=log_f, stderr=subprocess.STDOUT,
            timeout=TRAIN_TIMEOUT
        )
    return result.returncode == 0


def parse_run_log():
    scores = {}
    try:
        with open(RUN_LOG) as f:
            for line in f:
                m = re.match(r"^(hepatic_ci|death_ci|average_ci):\s+([\d.]+)", line)
                if m:
                    scores[m.group(1)] = float(m.group(2))
    except Exception:
        pass
    return scores


def tail_log(n=50):
    try:
        with open(RUN_LOG) as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception:
        return "(could not read log)"


def agent_turn(history):
    """
    This is where the AI agent would edit train.py.
    In autonomous mode, the agent runs before loop.py and modifies the file.
    For now, this function just verifies the file changed from the best commit.
    """
    if not os.path.exists(EXPERIMENT_FILE):
        print("❌ train.py missing!")
        return False
    return True


def main():
    history = load_history()

    # Record baseline commit on first run if we have git history
    if history.get("baseline_commit") is None and git_has_commits():
        history["baseline_commit"] = git_current_commit()
        save_history(history)

    exp_id = len(history["experiments"]) + 1
    print(f"\n{'='*60}")
    print(f"🧪 Experiment #{exp_id}")
    print(f"{'='*60}")

    # 1. Agent turn (assume train.py has been edited externally by agent)
    if not agent_turn(history):
        return

    # 2. Commit the current state before running
    if git_has_commits():
        parent = git_current_commit()
    else:
        parent = None

    subprocess.run(["git", "add", "-A"], capture_output=True)
    commit_msg = f"experiment #{exp_id}: pending"
    r = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
    if r.returncode != 0:
        print("⚠️  Nothing to commit (no changes?). Skipping experiment.")
        time.sleep(SLEEP_BETWEEN)
        return

    commit_hash = git_current_commit()

    # 3. Pre-screen
    if not run_simulate():
        print("❌ Pre-screen failed. Discarding...")
        status = "crash"
        scores = {}
        if parent:
            git_reset_hard(parent)
    else:
        # 4. Run experiment
        success = run_train()
        scores = parse_run_log()

        if not success or not scores or "average_ci" not in scores:
            print("❌ Experiment crashed or produced no scores.")
            print("Last 30 log lines:")
            print(tail_log(30))
            status = "crash"
            if parent:
                git_reset_hard(parent)
        else:
            avg = scores["average_ci"]
            best = history["best_score"]
            print(f"📊 Results: hepatic={scores.get('hepatic_ci', 0):.4f}, death={scores.get('death_ci', 0):.4f}, average={avg:.4f}")

            if avg > best:
                print(f"✅ NEW BEST! {avg:.4f} > {best:.4f} — keeping commit {commit_hash}")
                status = "keep"
                history["best_score"] = avg
                history["best_commit"] = commit_hash
                # Update commit message
                subprocess.run([
                    "git", "commit", "--amend", "-m",
                    f"experiment #{exp_id}: keep | avg={avg:.4f} hep={scores.get('hepatic_ci', 0):.4f} death={scores.get('death_ci', 0):.4f}"
                ], capture_output=True)
            else:
                print(f"❌ No improvement. {avg:.4f} <= {best:.4f} — discarding...")
                status = "discard"
                if parent:
                    git_reset_hard(parent)

    # 5. Log result
    record = {
        "id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "commit": commit_hash if status != "crash" else None,
        "hepatic_ci": scores.get("hepatic_ci", 0.0),
        "death_ci": scores.get("death_ci", 0.0),
        "average_ci": scores.get("average_ci", 0.0),
        "status": status
    }
    history["experiments"].append(record)
    save_history(history)
    print(f"📝 Logged to {HISTORY_FILE} | status={status}")


if __name__ == "__main__":
    # Continuous loop like karpathy/noqta pattern
    while True:
        try:
            main()
        except Exception as e:
            print(f"💥 Loop error: {e}")
        print(f"😴 Sleeping {SLEEP_BETWEEN}s...")
        time.sleep(SLEEP_BETWEEN)
