import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_data_and_catalogue.ipynb",
    "02_intent_and_kg.ipynb",
    "03_finetune_biencoder.ipynb",
    "04_retrieval_fusion_rerank.ipynb",
)


def main():
    os.chdir(ROOT)
    for filename in NOTEBOOKS:
        path = ROOT / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook["cells"][0]["cell_type"] != "code":
            raise ValueError(f"{filename} must begin with its setup code cell")
        namespace = {"__name__": "__main__"}
        print(f"\n==> Running {filename}")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            exec(compile(source, f"{filename}:cell-{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
