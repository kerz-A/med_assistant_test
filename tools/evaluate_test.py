"""Evaluate medical assistant test results against ground truth.

Metrics:
  TC  — Transcription Completeness (% of expected turns captured)
  SA  — Speaker Accuracy (% of correctly attributed roles)
  FER — Field Extraction Rate (% of expected fields filled)
  FVA — Field Value Accuracy (% of filled fields with correct values)
  DA  — Diagnosis Accuracy (ICD-10 code matches, binary)
  OQS — Overall Quality Score (weighted combination)

Usage:
    python evaluate_test.py results/01_cardiology.json
    python evaluate_test.py results/                     # all results
    python evaluate_test.py results/ --report report.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(__file__))
from test_scenarios import SCENARIOS

PROTOCOL_FIELDS = [
    ("patient_info", "full_name"),
    ("patient_info", "age"),
    ("patient_info", "gender"),
    ("exam_data", "complaints"),
    ("exam_data", "complaints_details"),
    ("exam_data", "anamnesis"),
    ("exam_data", "life_anamnesis"),
    ("exam_data", "allergies"),
    ("exam_data", "medications"),
    ("exam_data", "diagnosis"),
    ("exam_data", "treatment_plan"),
    ("exam_data", "patient_recommendations"),
    ("vitals", "height_cm"),
    ("vitals", "weight_kg"),
    ("vitals", "bmi"),
    ("vitals", "pulse"),
    ("vitals", "spo2"),
    ("vitals", "systolic_bp"),
]

NUMERIC_FIELDS = {"age", "height_cm", "weight_kg", "bmi", "pulse", "spo2"}
BP_FIELDS = {"systolic_bp"}

SAME_GENDER_SCENARIOS = {"03_neurology", "04_pulmonology", "08_urology", "09_pediatrics"}


@dataclass
class ScenarioMetrics:
    scenario_id: str
    scenario_name: str
    tc: float = 0.0
    sa: float = 0.0
    fer: float = 0.0
    fva: float = 0.0
    da: float = 0.0
    oqs: float = 0.0
    details: dict = field(default_factory=dict)


def fuzzy_match(actual: str, expected: str, threshold: float = 0.7) -> bool:
    if not actual or not expected:
        return False
    a = actual.lower().strip()
    e = expected.lower().strip()
    if e in a or a in e:
        return True
    return SequenceMatcher(None, a, e).ratio() >= threshold


def numeric_match(actual, expected, tolerance: float = 0.05) -> bool:
    try:
        a = float(actual)
        e = float(expected)
        if e == 0:
            return a == 0
        return abs(a - e) / abs(e) <= tolerance
    except (ValueError, TypeError):
        return False


def bp_match(actual: str, expected: str) -> bool:
    return str(actual).strip() == str(expected).strip()


# ============================================================
# Metric 1: Transcription Completeness (TC)
# ============================================================

def calc_tc(transcript: list[dict], expected_dialogue: list[tuple]) -> tuple[float, dict]:
    expected_count = len(expected_dialogue)
    actual_count = len(transcript)
    score = min(actual_count / expected_count, 1.0) * 100 if expected_count > 0 else 0.0
    return score, {"expected": expected_count, "actual": actual_count}


# ============================================================
# Metric 2: Speaker Accuracy (SA)
# ============================================================

def calc_sa(transcript: list[dict], expected_dialogue: list[tuple]) -> tuple[float, dict]:
    matched = 0
    total = 0
    mismatches = []
    used = set()

    for actual in transcript:
        best_idx, best_sim = -1, 0.0
        for i, (exp_role, exp_text) in enumerate(expected_dialogue):
            if i in used:
                continue
            sim = SequenceMatcher(None, actual["text"].lower(), exp_text.lower()).ratio()
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim > 0.3:
            used.add(best_idx)
            exp_role = expected_dialogue[best_idx][0]
            total += 1
            if actual["speaker"] == exp_role:
                matched += 1
            else:
                mismatches.append({
                    "text": actual["text"][:60],
                    "expected": exp_role,
                    "actual": actual["speaker"],
                })

    score = (matched / total * 100) if total > 0 else 0.0
    return score, {"matched": matched, "total": total, "mismatches": mismatches}


# ============================================================
# Metric 3: Field Extraction Rate (FER)
# ============================================================

def calc_fer(protocol: dict, expected: dict) -> tuple[float, dict]:
    expected_fields = []
    for section, fld in PROTOCOL_FIELDS:
        exp_val = expected.get(section, {}).get(fld)
        if exp_val is not None:
            expected_fields.append((section, fld))

    filled = 0
    missing = []
    for section, fld in expected_fields:
        actual_val = protocol.get(section, {}).get(fld)
        if actual_val is not None and actual_val != "" and actual_val != 0:
            filled += 1
        else:
            missing.append(f"{section}.{fld}")

    score = (filled / len(expected_fields) * 100) if expected_fields else 0.0
    return score, {"expected": len(expected_fields), "filled": filled, "missing": missing}


# ============================================================
# Metric 4: Field Value Accuracy (FVA)
# ============================================================

def calc_fva(protocol: dict, expected: dict) -> tuple[float, dict]:
    checked = 0
    correct = 0
    errors = []

    for section, fld in PROTOCOL_FIELDS:
        exp_val = expected.get(section, {}).get(fld)
        act_val = protocol.get(section, {}).get(fld)
        if exp_val is None or act_val is None:
            continue

        checked += 1
        match = False

        if fld in NUMERIC_FIELDS:
            match = numeric_match(act_val, exp_val)
        elif fld in BP_FIELDS:
            match = bp_match(act_val, exp_val)
        elif fld == "diagnosis":
            match = str(exp_val).upper() in str(act_val).upper()
        else:
            match = fuzzy_match(str(act_val), str(exp_val))

        if match:
            correct += 1
        else:
            errors.append({
                "field": f"{section}.{fld}",
                "expected": str(exp_val)[:80],
                "actual": str(act_val)[:80],
            })

    score = (correct / checked * 100) if checked > 0 else 0.0
    return score, {"checked": checked, "correct": correct, "errors": errors}


# ============================================================
# Metric 5: Diagnosis Accuracy (DA)
# ============================================================

def calc_da(protocol: dict, expected: dict) -> tuple[float, dict]:
    expected_code = expected.get("exam_data", {}).get("diagnosis", "")
    actual_diag = protocol.get("exam_data", {}).get("diagnosis", "") or ""

    if not expected_code:
        return 1.0, {"note": "no expected code"}

    found = expected_code.upper() in actual_diag.upper()
    return (1.0 if found else 0.0), {
        "expected_code": expected_code,
        "actual": actual_diag[:120],
        "match": found,
    }


# ============================================================
# Metric 6: Overall Quality Score (OQS)
# ============================================================

def calc_oqs(tc, sa, fer, fva, da) -> float:
    return 0.20 * tc + 0.20 * sa + 0.25 * fer + 0.25 * fva + 0.10 * (da * 100)


# ============================================================
# Evaluation
# ============================================================

def evaluate_result(result: dict) -> ScenarioMetrics:
    scenario_id = result["scenario_id"]
    scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    transcript = result.get("transcript", [])
    protocol = result.get("protocol", {})
    expected = scenario["expected_protocol"]
    full_dialogue = scenario["calibration_dialogue"] + scenario["exam_dialogue"]

    tc, tc_d = calc_tc(transcript, full_dialogue)
    sa, sa_d = calc_sa(transcript, full_dialogue)
    fer, fer_d = calc_fer(protocol, expected)
    fva, fva_d = calc_fva(protocol, expected)
    da, da_d = calc_da(protocol, expected)
    oqs = calc_oqs(tc, sa, fer, fva, da)

    return ScenarioMetrics(
        scenario_id=scenario_id,
        scenario_name=scenario["name"],
        tc=round(tc, 1), sa=round(sa, 1),
        fer=round(fer, 1), fva=round(fva, 1),
        da=da, oqs=round(oqs, 1),
        details={"TC": tc_d, "SA": sa_d, "FER": fer_d, "FVA": fva_d, "DA": da_d},
    )


def print_report(metrics_list: list[ScenarioMetrics]):
    print(f"\n{'='*95}")
    print(f"{'Scenario':<40} {'TC':>6} {'SA':>6} {'FER':>6} {'FVA':>6} {'DA':>4} {'OQS':>6}")
    print(f"{'-'*95}")

    for m in metrics_list:
        da_str = "YES" if m.da == 1.0 else "NO"
        print(f"{m.scenario_name[:39]:<40} {m.tc:>5.1f}% {m.sa:>5.1f}% {m.fer:>5.1f}% {m.fva:>5.1f}% {da_str:>4} {m.oqs:>5.1f}%")

    print(f"{'-'*95}")
    n = len(metrics_list)
    if n > 0:
        avg = lambda attr: sum(getattr(m, attr) for m in metrics_list) / n
        da_pct = sum(m.da for m in metrics_list) / n
        print(f"{'AVERAGE':<40} {avg('tc'):>5.1f}% {avg('sa'):>5.1f}% {avg('fer'):>5.1f}% {avg('fva'):>5.1f}% {da_pct:>3.0%} {avg('oqs'):>5.1f}%")
    print(f"{'='*95}")

    same = [m for m in metrics_list if m.scenario_id in SAME_GENDER_SCENARIOS]
    diff = [m for m in metrics_list if m.scenario_id not in SAME_GENDER_SCENARIOS]
    if same and diff:
        sa_same = sum(m.sa for m in same) / len(same)
        sa_diff = sum(m.sa for m in diff) / len(diff)
        print(f"\nSpeaker Accuracy by gender:")
        print(f"  Same gender (HARD):      {sa_same:.1f}%")
        print(f"  Different gender (EASY):  {sa_diff:.1f}%")

    # Print field-level errors
    all_errors = []
    for m in metrics_list:
        for e in m.details.get("FVA", {}).get("errors", []):
            all_errors.append({"scenario": m.scenario_id, **e})
    if all_errors:
        print(f"\nField Value Errors ({len(all_errors)}):")
        for e in all_errors[:20]:
            print(f"  [{e['scenario']}] {e['field']}: expected='{e['expected'][:40]}' got='{e['actual'][:40]}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate test results against ground truth")
    parser.add_argument("path", help="Result JSON file or directory with results")
    parser.add_argument("--report", help="Save report as JSON")
    args = parser.parse_args()

    result_files = []
    if os.path.isdir(args.path):
        for f in sorted(os.listdir(args.path)):
            if f.endswith(".json"):
                result_files.append(os.path.join(args.path, f))
    else:
        result_files = [args.path]

    if not result_files:
        print("No result files found")
        return

    metrics_list = []
    for path in result_files:
        try:
            with open(path) as f:
                result = json.load(f)
            m = evaluate_result(result)
            metrics_list.append(m)
        except Exception as e:
            print(f"Error evaluating {path}: {e}")

    print_report(metrics_list)

    if args.report:
        report = {
            "scenarios": [
                {
                    "id": m.scenario_id, "name": m.scenario_name,
                    "TC": m.tc, "SA": m.sa, "FER": m.fer, "FVA": m.fva,
                    "DA": m.da, "OQS": m.oqs, "details": m.details,
                }
                for m in metrics_list
            ],
        }
        with open(args.report, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {args.report}")


if __name__ == "__main__":
    main()
