import os
import sys
import argparse
from git import Repo

def auto_checkpoint(message="Auto update from pipeline", repo_dir="."):
    try:
        repo = Repo(repo_dir)
        repo.git.add(all=True)
        if repo.is_dirty():
            repo.index.commit(message)
            origin = repo.remote(name='origin')
            origin.push()
            print(f"✅ Git sync successful: {message}")
        else:
            print("ℹ️ No changes detected to commit.")
    except Exception as e:
        print(f"⚠️ Git sync warning: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="Pipeline auto-checkpoint")
    args = parser.parse_args()
    auto_checkpoint(args.message)
