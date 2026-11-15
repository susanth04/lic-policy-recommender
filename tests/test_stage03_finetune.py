import json
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def finetune_status():
    notebook = nbformat.read(ROOT / "03_finetune_biencoder.ipynb", as_version=4)
    state = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(ROOT / "03_finetune_biencoder.ipynb"), "exec"), state)
    return state["manifest"]


def test_finetuning_waits_for_real_grades(finetune_status):
    assert finetune_status["status"] == "SKIPPED_NO_HUMAN_GRADED_LABELS"
    assert finetune_status["human_graded_rows"] == 0
    assert finetune_status["trained_on_synthetic_labels"] is False
    assert finetune_status["experiments"] == {}


def test_policy_holdout_and_group_splits_are_disjoint(finetune_status):
    metadata = json.loads((ROOT / "artifacts" / "model" / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "artifacts" / "splits.json").read_text(encoding="utf-8"))
    groups = splits["group_split"]["groups"]

    assert not (set(groups["train"]) & set(groups["validation"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert set(splits["policy_held_out"]["training_policy_ids"]).isdisjoint(
        splits["policy_held_out"]["held_out_policy_ids"]
    )
    assert metadata["cache_directory"] == "artifacts/model"
