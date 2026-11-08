from __future__ import annotations

from agents.etl.governance_manager import (
    apply_governance_to_options,
    get_default_governance_rules,
    load_governance_rules,
    reset_governance_rules,
    save_governance_rules,
)
from agents.etl.models import ETLOptions


def test_governance_defaults():
    rules = get_default_governance_rules()
    assert "global_settings" in rules
    assert "range_rules" in rules
    assert rules["global_settings"]["drop_column_missing_pct"] == 80.0
    assert len(rules["range_rules"]) >= 5


def test_governance_save_and_load(tmp_path, monkeypatch):
    test_file = tmp_path / "gov_test.json"
    monkeypatch.setattr("agents.etl.governance_manager.GOVERNANCE_CONFIG_FILE", test_file)

    custom_rules = get_default_governance_rules()
    custom_rules["global_settings"]["drop_column_missing_pct"] = 75.0
    custom_rules["range_rules"].append({
        "column": "custom_metric",
        "min": 10,
        "max": 100,
        "fix": "clamp",
        "description": "Test rule"
    })

    save_governance_rules(custom_rules)
    loaded = load_governance_rules()
    assert loaded["global_settings"]["drop_column_missing_pct"] == 75.0
    assert any(r["column"] == "custom_metric" for r in loaded["range_rules"])

    # Test reset
    reset_governance_rules()
    reloaded = load_governance_rules()
    assert reloaded["global_settings"]["drop_column_missing_pct"] == 80.0


def test_apply_governance_to_options(tmp_path, monkeypatch):
    test_file = tmp_path / "gov_test.json"
    monkeypatch.setattr("agents.etl.governance_manager.GOVERNANCE_CONFIG_FILE", test_file)

    custom_rules = get_default_governance_rules()
    custom_rules["global_settings"]["drop_column_missing_pct"] = 65.0
    custom_rules["global_settings"]["pii_policy"] = "drop"
    save_governance_rules(custom_rules)

    base_options = ETLOptions()
    merged = apply_governance_to_options(base_options)

    assert merged.drop_column_missing_pct == 65.0
    assert merged.pii_policy == "drop"
    assert len(merged.range_rules) >= len(custom_rules["range_rules"])
