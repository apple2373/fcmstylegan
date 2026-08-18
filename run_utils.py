"""Shared utilities for recording training-run provenance."""

import os
import subprocess


def save_git_metadata(run_dir):
    """Save Git status, commit hash, and diff for a training run."""
    try:
        git_status = subprocess.check_output(
            ["git", "status"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        with open(os.path.join(run_dir, "git_status.txt"), "w", encoding="utf-8") as file:
            file.write(git_status)

        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        with open(os.path.join(run_dir, "git_hash.txt"), "w", encoding="utf-8") as file:
            file.write(git_hash.strip())

        git_diff = subprocess.check_output(
            ["git", "diff"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        with open(os.path.join(run_dir, "git_diff.txt"), "w", encoding="utf-8") as file:
            file.write(git_diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        with open(os.path.join(run_dir, "git_info_error.txt"), "w", encoding="utf-8") as file:
            file.write("Git information could not be retrieved. (Not a git repository or git not installed)\n")
