import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path.cwd() / "src"))
from models import build_model

def main():
    b = torch.load("submission/v17_model_weights.pt", map_location="cpu", weights_only=False)
    print("Members in v17 bundle:", len(b["members"]))
    for m in b["members"]:
        model = build_model(m["arch"], pretrained=False)
        missing, unexpected = model.load_state_dict(m["state_dict"], strict=True)
        print(f"  {m['name']} ({m['arch']}): STRICT LOAD OK (0 missing, 0 unexpected)")

if __name__ == "__main__":
    main()
