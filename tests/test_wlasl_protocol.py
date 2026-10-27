import runpy
from pathlib import Path

import pytest


functions = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_wlasl_training.py"))
select_rows = functions["select_rows"]
split_classes = functions["split_classes"]


def rows_and_audit():
    rows = [{"video_id": f"c{c}_{v}", "class_key": f"ASL:{c}", "gloss": f"g{c}"}
            for c in range(30) for v in range(4)]
    # Two linked classes and a within-class duplicate; c0_1 is unusable.
    groups = [{"source_hash": "shared", "videos": [rows[0], rows[4]]},
              {"source_hash": "within", "videos": [rows[8], rows[9]]}]
    audit = {"checked_videos": len(rows), "cache_errors": [],
             "all_invalid_by_variant": {"common": [rows[1]]}, "duplicate_source_groups": groups}
    return rows, audit


def test_global_dedup_invalid_removal_and_class_threshold():
    rows, audit = rows_and_audit()
    selected, rejected = select_rows(rows, audit, 3)
    ids = {row["video_id"] for row in selected}
    assert "c0_1" not in ids
    assert "c0_0" in ids and "c1_0" not in ids  # Preserve the rarer class.
    assert len(ids & {"c2_0", "c2_1"}) == 1
    assert len(selected) + len(rejected) == len(rows)
    # Raising the threshold excludes all remaining members of undersized classes.
    selected, _ = select_rows(rows, audit, 4)
    assert not {"ASL:0", "ASL:1", "ASL:2"} & {row["class_key"] for row in selected}


def test_deterministic_class_split_keeps_linked_classes_together():
    rows, audit = rows_and_audit()
    assignments = split_classes(rows, audit["duplicate_source_groups"], 42)
    assert assignments == split_classes(list(reversed(rows)), audit["duplicate_source_groups"], 42)
    assert assignments["ASL:0"] == assignments["ASL:1"]
    assert set(assignments) == {row["class_key"] for row in rows}
    assert list(assignments.values()).count("train") == 18
    assert list(assignments.values()).count("val") == 6
    assert list(assignments.values()).count("test") == 6


def test_ratio_rounding_assigns_every_class_once():
    rows = [{"class_key": f"ASL:{c}"} for c in range(12)]
    assignments = split_classes(rows, [], 42)
    assert len(assignments) == 12
    assert list(assignments.values()).count("train") == 7
    assert list(assignments.values()).count("val") == 2
    assert list(assignments.values()).count("test") == 3


def test_stale_or_inconsistent_audit_is_rejected():
    rows, audit = rows_and_audit()
    audit["checked_videos"] -= 1
    with pytest.raises(ValueError, match="cover"):
        select_rows(rows, audit, 3)
    audit["checked_videos"] += 1
    audit["duplicate_source_groups"][0]["videos"][0] = {"video_id": "missing", "class_key": "ASL:0"}
    with pytest.raises(ValueError, match="disagree"):
        select_rows(rows, audit, 3)
