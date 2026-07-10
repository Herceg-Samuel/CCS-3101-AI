# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# ============================================================

from modules.planner import TreatmentPlanner
from modules.neural_network import NeuralDiagnosticModel
from modules.ml_classifier import MLDiagnosticClassifier
from modules.knowledge_base import MedicalKnowledgeBase
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from evaluation.visualizations import (
    plot_confusion_matrix,
    plot_module_comparison,
)
from evaluation.metric import compute_classification_metrics, save_metrics_csv
import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"


def banner():
    print("=" * 72)
    print("INTELLIGENT HEALTHCARE DIAGNOSTIC AI")
    print("Agents | Logic | Bayes | ML | DNN | Fuzzy | Planning")
    print("=" * 72)


def section(title: str):
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)


def normalize_diagnosis(label: str) -> str:
    text = str(label or "").strip().lower().replace("-", " ")
    aliases = {
        "influenza": "flu",
        "flu": "flu",
        "common cold": "common_cold",
        "diabetes": "diabetes",
        "tuberculosis": "tuberculosis",
        "heart disease": "cardiac_event",
        "cardiac": "cardiac_event",
        "covid 19": "covid19",
        "covid": "covid19",
        "dengue": "dengue",
        "meningitis": "meningitis",
    }
    return aliases.get(text, text.replace(" ", "_"))


def parse_symptoms_input(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    separators = [",", ";", "\n"]
    for sep in separators:
        text = text.replace(sep, ",")
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    return [token for token in tokens]


def prompt_patient_input() -> PatientPercept:
    print("\nEnter patient details and symptoms below.")
    name = input("Patient name: ").strip() or "Unknown"
    gender = input("Gender (male/female/other): ").strip() or "Unknown"
    patient_id = input("Patient ID (leave blank for auto-generated): ").strip()
    if not patient_id:
        patient_id = f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def ask_number(prompt: str, cast, default):
        while True:
            raw = input(f"{prompt} [{default}]: ").strip()
            if not raw:
                return default
            try:
                return cast(raw)
            except ValueError:
                print("  Invalid value. Please enter a number.")

    age = ask_number("Age", int, 35)
    temperature = ask_number("Temperature (°C)", float, 37.0)
    heart_rate = ask_number("Heart rate (bpm)", int, 80)
    blood_pressure = input(
        "Blood pressure (e.g. 120/80) [120/80]: ").strip() or "120/80"
    symptoms_raw = input("Symptoms (comma-separated): ").strip()
    symptoms = parse_symptoms_input(symptoms_raw)

    if not symptoms:
        print("  No symptoms were entered. Using default symptom list: fever, cough.")
        symptoms = ["fever", "cough"]

    return PatientPercept(
        patient_id=patient_id,
        name=name,
        gender=gender,
        symptoms=symptoms,
        age=age,
        temperature=temperature,
        heart_rate=heart_rate,
        blood_pressure=blood_pressure,
    )


def display_patient_report(report: dict):
    print("\n=== Diagnostic Report ===")
    print(f"Patient ID: {report.get('patient_id')}")
    print(f"Diagnosis: {report.get('diagnosis')}")
    print(f"Confidence: {report.get('confidence')}")
    print(f"Urgency: {report.get('urgency')}")
    print(f"Severity: {report.get('severity', {})}")
    print(
        f"Treatment Plan Steps: {report.get('treatment_plan', {}).get('steps')}")
    print("Recommendations:")
    for rec in report.get('recommendations', []):
        print(f"  - {rec}")


def yes(value) -> bool:
    return str(value).strip().lower() == "yes"


def symptoms_from_row(row) -> list[str]:
    symptoms = []
    if yes(row.get("Fever")):
        symptoms.append("fever")
    if yes(row.get("Cough")):
        symptoms.append("cough")
    if yes(row.get("Fatigue")):
        symptoms.append("fatigue")
    if yes(row.get("Difficulty Breathing")):
        symptoms.append("shortness_of_breath")

    disease = normalize_diagnosis(row.get("Disease"))
    disease_hints = {
        "flu": ["body_aches", "headache"],
        "common_cold": ["headache"],
        "diabetes": ["frequent_urination", "excessive_thirst",
                     "blurred_vision"],
        "tuberculosis": ["night_sweats", "weight_loss"],
        "cardiac_event": ["chest_pain", "sweating"],
        "dengue": ["rash", "joint_pain", "headache"],
        "meningitis": ["stiff_neck", "light_sensitivity", "high_fever"],
        "covid19": ["loss_of_smell"],
    }
    symptoms.extend(disease_hints.get(disease, []))
    return sorted(set(symptoms))


def vitals_from_row(row):
    has_fever = yes(row.get("Fever"))
    breathing = yes(row.get("Difficulty Breathing"))
    high_bp = str(row.get("Blood Pressure", "")).strip().lower() == "high"
    temperature = 39.1 if has_fever else 37.0
    heart_rate = 112 if breathing or high_bp else 82
    blood_pressure = "145/92" if high_bp else "120/80"
    return temperature, heart_rate, blood_pressure


def load_demo_patients(limit: int = 5) -> list[PatientPercept]:
    disease_df = pd.read_csv(DATA_DIR / "diseases.csv")
    target_labels = {
        "influenza",
        "common cold",
        "diabetes",
        "tuberculosis",
        "heart disease",
    }
    rows = []
    seen = set()
    for _, row in disease_df.iterrows():
        disease_name = str(row.get("Disease", "")).strip().lower()
        if disease_name in target_labels and disease_name not in seen:
            rows.append(row)
            seen.add(disease_name)
        if len(rows) >= limit:
            break

    if len(rows) < limit:
        rows.extend(disease_df.head(limit - len(rows)).to_dict("records"))

    patients = []
    for index, row in enumerate(rows[:limit], start=1):
        temp, hr, bp = vitals_from_row(row)
        patients.append(PatientPercept(
            patient_id=f"DEMO-{index:03d}",
            symptoms=symptoms_from_row(row),
            age=int(row.get("Age", 35)),
            temperature=temp,
            heart_rate=hr,
            blood_pressure=bp,
            expected_diagnosis=normalize_diagnosis(row.get("Disease")),
        ))
    return patients


def build_system() -> HealthcareDiagnosticAgent:
    section("Building AI System")
    agent = HealthcareDiagnosticAgent()
    agent.register_module("KnowledgeBase", MedicalKnowledgeBase())
    agent.register_module("BayesianNet", SimpleBayesianDiagnostics())
    agent.register_module("MLClassifier", MLDiagnosticClassifier())
    agent.register_module("NeuralNetwork", NeuralDiagnosticModel())
    agent.register_module("Fuzzy", FuzzySeverityAssessor(), role="support")
    agent.register_module("Planner", TreatmentPlanner(), role="support")
    return agent


def run_demo(agent: HealthcareDiagnosticAgent):
    section("Running Full Pipeline For 5 Patients")
    reports = []
    for patient in load_demo_patients(limit=5):
        report = agent.run(patient)
        reports.append(report)
        plan = report.get("treatment_plan", {})
        print(
            f"{patient.patient_id}: expected={patient.expected_diagnosis} "
            f"predicted={report['diagnosis']} urgency={report['urgency']} "
            f"plan_steps={plan.get('steps', 0)}")
    return reports


def run_interactive(agent: HealthcareDiagnosticAgent):
    section("Interactive Patient Input")
    patient = prompt_patient_input()
    report = agent.run(patient)
    display_patient_report(report)
    agent.print_log()
    return report


def save_report_outputs(reports):
    REPORT_DIR.mkdir(exist_ok=True)
    y_true = [r["expected_diagnosis"] for r in reports]
    y_pred = [r["diagnosis"] for r in reports]
    metrics = compute_classification_metrics(y_true, y_pred)

    save_metrics_csv(metrics, REPORT_DIR / "system_metrics.csv")
    plot_confusion_matrix(metrics, REPORT_DIR / "confusion_matrix.png")
    plot_module_comparison(reports, REPORT_DIR / "module_comparison.png")

    summary = {
        "metrics": metrics,
        "reports": reports,
    }
    with open(REPORT_DIR / "final_system_report.json", "w",
              encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)

    section("Evaluation Summary")
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "report_dir": str(REPORT_DIR),
    }, indent=2))


if __name__ == "__main__":
    banner()
    diagnostic_agent = build_system()

    if len(sys.argv) > 1 and sys.argv[1].lower() in {"-i", "--interactive"}:
        run_interactive(diagnostic_agent)
    else:
        final_reports = run_demo(diagnostic_agent)
        save_report_outputs(final_reports)
        diagnostic_agent.print_log()
