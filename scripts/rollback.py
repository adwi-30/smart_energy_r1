import os, shutil, subprocess, json

POLICY_DIR = "policies"
BACKUP_DIR = "policies/backup"
LOG_FILE   = "experiments/rollback_log.json"

def rollback_to_tag(tag: str):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    # Backup current policy
    for f in os.listdir(POLICY_DIR):
        if f.endswith(".pkl"):
            shutil.copy(os.path.join(POLICY_DIR, f),
                        os.path.join(BACKUP_DIR, f + ".bak"))
    # Checkout policy from tag
    subprocess.run(["git", "checkout", tag, "--", POLICY_DIR], check=True)
    # Log rollback
    log = {"rolled_back_to": tag, "status": "success"}
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Rolled back to {tag} successfully.")

if __name__ == "__main__":
    import sys
    rollback_to_tag(sys.argv[1] if len(sys.argv) > 1 else "exp-qlearning-1")
