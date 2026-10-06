import sys
from pathlib import Path

MILESTONE_4 = Path(__file__).resolve().parent.parent
for folder in ("retraining", "dashboard", "workload"):
    sys.path.insert(0, str(MILESTONE_4 / folder))
