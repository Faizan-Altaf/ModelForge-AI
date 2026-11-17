from pathlib import Path
import shutil
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

CHAMPION_DIR = ROOT / "models" / "champion"
ACTIVE = CHAMPION_DIR / "champion.pth"

if len(sys.argv) > 1:
    backup = Path(sys.argv[1])
else:
    backups = sorted(
        CHAMPION_DIR.glob("champion_backup_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not backups:
        raise FileNotFoundError("No Champion backup found.")

    backup = backups[0]

if not backup.exists():
    raise FileNotFoundError(backup)

if ACTIVE.exists():

    snapshot = CHAMPION_DIR / (
        "champion_before_rollback_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pth"
    )

    shutil.copy2(ACTIVE, snapshot)
    print("Current Champion snapshot:", snapshot)

shutil.copy2(backup, ACTIVE)

print("Rollback completed.")
print("Restored model:", backup)
print("Active model:", ACTIVE)
