import os
import re
from pathlib import Path

ROOT = Path(r"d:\hj\Y-3 S-1\POD-Sim2Real")

CONFIG_DIRS = [
    ROOT / "yaml_main",
    ROOT / "yaml_main_v2",
    ROOT / "yaml_operators",
    ROOT / "yaml_ablation",
    ROOT / "configs",
]

def update_yaml_file(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text
    
    # Update lr in optimizer section
    # Match "lr: <number>"
    text = re.sub(r"(^\s*lr:\s*)[0-9\.eE-]+", r"\g<1>0.0002", text, flags=re.MULTILINE)
    
    # Update pretrain_epochs: <number>
    text = re.sub(r"(^\s*pretrain_epochs:\s*)\d+", r"\g<1>200", text, flags=re.MULTILINE)
    
    # Update finetune_epochs: <number>
    text = re.sub(r"(^\s*finetune_epochs:\s*)\d+", r"\g<1>200", text, flags=re.MULTILINE)
    
    # Update standalone epochs: <number> (if not preceded by pretrain_ or finetune_)
    text = re.sub(r"(^\s*epochs:\s*)\d+", r"\g<1>200", text, flags=re.MULTILINE)
    
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"Updated: {path.relative_to(ROOT)}")
    else:
        print(f"No change: {path.relative_to(ROOT)}")

count = 0
for d in CONFIG_DIRS:
    if not d.exists():
        continue
    for yf in sorted(d.glob("*.yaml")):
        update_yaml_file(yf)
        count += 1

print(f"\nTotal YAML files checked/updated: {count}")
