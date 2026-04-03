"""Evaluate medical assistant test results against ground truth.

Metrics:
  TC  — Transcription Completeness (% of expected turns captured)
  SA  — Speaker Accuracy (% of correctly attributed roles)
  FER — Field Extraction Rate (% of expected fields filled)
  FVA — Field Value Accuracy (% of filled fields with correct values)
  DA  — Diagnosis Accuracy (ICD-10 code matches, binary)
  OQS — Overall Quality Score (weighted combination)
  CDS — Clinical Decision Support Score (0-5, quality of consultation assessment)

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
class CDSMetrics:
    """Clinical Decision Support metrics."""
    overall_score: float = 0.0       # 0-5 scale
    criteria_completed: int = 0      # how many of 11 criteria scored > 0
    criteria_total: int = 11
    quality_criteria: dict = field(default_factory=dict)
    dialogue_analytics: dict = field(default_factory=dict)


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
    cds: CDSMetrics = field(default_factory=CDSMetrics)
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

def _normalize_dialogue(dialogue: list) -> list[tuple]:
    """Convert dialogue to list of (role, text) tuples regardless of input format."""
    result = []
    for turn in dialogue:
        if isinstance(turn, dict):
            result.append((turn["role"], turn["text"]))
        else:
            result.append((turn[0], turn[1]))
    return result


def calc_tc(transcript: list[dict], expected_dialogue: list) -> tuple[float, dict]:
    expected_dialogue = _normalize_dialogue(expected_dialogue)
    expected_count = len(expected_dialogue)
    actual_count = len(transcript)
    score = min(actual_count / expected_count, 1.0) * 100 if expected_count > 0 else 0.0
    return score, {"expected": expected_count, "actual": actual_count}


# ============================================================
# Metric 2: Speaker Accuracy (SA)
# ============================================================

def calc_sa(transcript: list[dict], expected_dialogue: list) -> tuple[float, dict]:
    expected_dialogue = _normalize_dialogue(expected_dialogue)
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
# Metric 7: Clinical Decision Support (CDS)
# ============================================================

QUALITY_CRITERIA_FIELDS = [
    "greeting_and_contact", "conversation_structure", "needs_identification",
    "current_complaints_identification", "disease_history", "general_medical_history",
    "medication_history", "family_history", "prevention_and_risk_control",
    "treatment_planning", "visit_closure",
]

DIALOGUE_ANALYTICS_FIELDS = [
    "doctor_showed_empathy", "doctor_interrupted_patient", "patient_asked_questions",
    "doctor_used_medical_jargon", "doctor_confirmed_understanding", "lifestyle_discussed",
    "allergies_discussed", "shared_decision_making", "patient_compliance_assessment",
    "doctor_pacing",
]


def extract_cds(protocol: dict) -> CDSMetrics:
    """Extract Clinical Decision Support metrics from protocol."""
    cds_data = protocol.get("clinical_decision_support", {})
    if not cds_data:
        return CDSMetrics()

    qc = cds_data.get("quality_criteria", {})
    da = cds_data.get("dialogue_analytics", {})
    eq = cds_data.get("examination_quality", {})

    # Recalculate overall score from criteria (in case examination_quality is missing)
    scores = [qc.get(f, 0) for f in QUALITY_CRITERIA_FIELDS]
    total = sum(scores)
    max_total = 11 * 2  # 22
    criteria_completed = sum(1 for s in scores if s > 0)
    overall_score = eq.get("overall_score", round((total / max_total) * 5, 1) if max_total else 0.0)

    return CDSMetrics(
        overall_score=overall_score,
        criteria_completed=criteria_completed,
        criteria_total=11,
        quality_criteria=qc,
        dialogue_analytics=da,
    )


# ============================================================
# Evaluation
# ============================================================

def _normalize_expected(scenario_expected: dict, scenario: dict) -> dict:
    """Convert flat expected_protocol to nested format matching actual protocol structure."""
    exp = scenario_expected

    # Extract full_name from calibration dialogue (patient's first line)
    full_name = None
    cal = scenario.get("calibration_dialogue", [])
    for turn in cal:
        role = turn["role"] if isinstance(turn, dict) else turn[0]
        text = turn["text"] if isinstance(turn, dict) else turn[1]
        if role == "patient":
            full_name = text.split(",")[0].split(".")[0].strip()
            # Remove common prefixes
            for prefix in ["Здравствуйте", "Добрый день", "Меня зовут", "Ребёнка зовут"]:
                full_name = full_name.replace(prefix, "").strip()
            if full_name.startswith("."):
                full_name = full_name[1:].strip()
            break

    return {
        "patient_info": {
            "full_name": full_name,
            "age": exp.get("age"),
            "gender": exp.get("gender"),
        },
        "exam_data": {
            "complaints": exp.get("complaints"),
            "complaints_details": exp.get("complaints_details"),
            "anamnesis": exp.get("anamnesis"),
            "life_anamnesis": exp.get("life_anamnesis"),
            "allergies": exp.get("allergies"),
            "medications": exp.get("medications"),
            "diagnosis": exp.get("diagnosis"),
            "treatment_plan": exp.get("treatment_plan"),
            "patient_recommendations": exp.get("patient_recommendations"),
        },
        "vitals": {
            "height_cm": float(exp["height"]) if exp.get("height") is not None else None,
            "weight_kg": float(exp["weight"]) if exp.get("weight") is not None else None,
            "bmi": exp.get("bmi"),
            "pulse": float(exp["pulse"]) if exp.get("pulse") is not None else None,
            "spo2": float(exp["spo2"]) if exp.get("spo2") is not None else None,
            "systolic_bp": exp.get("bp"),
        },
    }


def evaluate_result(result: dict) -> ScenarioMetrics:
    scenario_id = result["scenario_id"]
    scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    transcript = result.get("transcript", [])
    protocol = result.get("protocol", {})
    expected = _normalize_expected(scenario["expected_protocol"], scenario)
    full_dialogue = scenario["calibration_dialogue"] + scenario["exam_dialogue"]

    tc, tc_d = calc_tc(transcript, full_dialogue)
    sa, sa_d = calc_sa(transcript, full_dialogue)
    fer, fer_d = calc_fer(protocol, expected)
    fva, fva_d = calc_fva(protocol, expected)
    da, da_d = calc_da(protocol, expected)
    oqs = calc_oqs(tc, sa, fer, fva, da)
    cds = extract_cds(protocol)

    return ScenarioMetrics(
        scenario_id=scenario_id,
        scenario_name=scenario.get("name", scenario_id),
        tc=round(tc, 1), sa=round(sa, 1),
        fer=round(fer, 1), fva=round(fva, 1),
        da=da, oqs=round(oqs, 1), cds=cds,
        details={"TC": tc_d, "SA": sa_d, "FER": fer_d, "FVA": fva_d, "DA": da_d},
    )


def print_report(metrics_list: list[ScenarioMetrics]):
    print(f"\n{'='*105}")
    print(f"{'Scenario':<40} {'TC':>6} {'SA':>6} {'FER':>6} {'FVA':>6} {'DA':>4} {'OQS':>6} {'CDS':>7}")
    print(f"{'-'*105}")

    for m in metrics_list:
        da_str = "YES" if m.da == 1.0 else "NO"
        cds_str = f"{m.cds.overall_score:.1f}/5" if m.cds.overall_score > 0 else "  —"
        print(f"{m.scenario_name[:39]:<40} {m.tc:>5.1f}% {m.sa:>5.1f}% {m.fer:>5.1f}% {m.fva:>5.1f}% {da_str:>4} {m.oqs:>5.1f}% {cds_str:>7}")

    print(f"{'-'*105}")
    n = len(metrics_list)
    if n > 0:
        avg = lambda attr: sum(getattr(m, attr) for m in metrics_list) / n
        da_pct = sum(m.da for m in metrics_list) / n
        cds_with_data = [m for m in metrics_list if m.cds.overall_score > 0]
        cds_avg_str = f"{sum(m.cds.overall_score for m in cds_with_data) / len(cds_with_data):.1f}/5" if cds_with_data else "  —"
        print(f"{'AVERAGE':<40} {avg('tc'):>5.1f}% {avg('sa'):>5.1f}% {avg('fer'):>5.1f}% {avg('fva'):>5.1f}% {da_pct:>3.0%} {avg('oqs'):>5.1f}% {cds_avg_str:>7}")
    print(f"{'='*105}")

    same = [m for m in metrics_list if m.scenario_id in SAME_GENDER_SCENARIOS]
    diff = [m for m in metrics_list if m.scenario_id not in SAME_GENDER_SCENARIOS]
    if same and diff:
        sa_same = sum(m.sa for m in same) / len(same)
        sa_diff = sum(m.sa for m in diff) / len(diff)
        print(f"\nSpeaker Accuracy by gender:")
        print(f"  Same gender (HARD):      {sa_same:.1f}%")
        print(f"  Different gender (EASY):  {sa_diff:.1f}%")

    # Print CDS details if available
    cds_scenarios = [m for m in metrics_list if m.cds.overall_score > 0]
    if cds_scenarios:
        print(f"\nClinical Decision Support Details:")
        print(f"  {'Scenario':<35} {'Score':>5} {'Criteria':>10} {'Interrupted':>12}")
        print(f"  {'-'*65}")
        for m in cds_scenarios:
            interrupted = m.cds.dialogue_analytics.get("doctor_interrupted_patient", "—")
            print(f"  {m.scenario_name[:34]:<35} {m.cds.overall_score:>4.1f}  "
                  f"{m.cds.criteria_completed:>4}/{m.cds.criteria_total:<4} "
                  f"{interrupted:>10}")
        print()
        # Show quality criteria distribution
        all_scores = []
        for m in cds_scenarios:
            for f in QUALITY_CRITERIA_FIELDS:
                all_scores.append(m.cds.quality_criteria.get(f, 0))
        if all_scores:
            count_0 = all_scores.count(0)
            count_1 = all_scores.count(1)
            count_2 = all_scores.count(2)
            total = len(all_scores)
            print(f"  Score distribution: 0={count_0} ({count_0/total:.0%})  "
                  f"1={count_1} ({count_1/total:.0%})  "
                  f"2={count_2} ({count_2/total:.0%})")

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
                    "DA": m.da, "OQS": m.oqs,
                    "CDS": {
                        "overall_score": m.cds.overall_score,
                        "criteria_completed": m.cds.criteria_completed,
                        "criteria_total": m.cds.criteria_total,
                        "quality_criteria": m.cds.quality_criteria,
                        "dialogue_analytics": m.cds.dialogue_analytics,
                    },
                    "details": m.details,
                }
                for m in metrics_list
            ],
        }
        with open(args.report, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {args.report}")


if __name__ == "__main__":
    main()
