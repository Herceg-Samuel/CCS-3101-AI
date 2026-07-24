# ============================================================
# WEB SERVER
# Flask front door for the Intelligent Healthcare Diagnostic
# Agent. Wraps the existing agent/module pipeline (app.py,
# modules/*) with a JSON API consumed by static/js/app.js.
# ============================================================

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, render_template, send_from_directory

from modules.agent import PatientPercept
from app import (
    build_system,
    load_demo_patients,
    parse_symptoms_input,
    normalize_diagnosis,
    run_demo,
    save_report_outputs,
)

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "images"

app = Flask(__name__)

# Canonical symptom vocabulary understood by the vector-based
# models (ML ensemble + neural network). Free-text symptoms are
# still accepted for the logic/Bayesian modules but only these
# map onto the trained feature vectors.
SYMPTOM_CATALOG = [
    ("fever", "Fever"),
    ("cough", "Cough"),
    ("fatigue", "Fatigue"),
    ("headache", "Headache"),
    ("body_aches", "Body aches"),
    ("loss_of_smell", "Loss of smell"),
    ("chest_pain", "Chest pain"),
    ("rash", "Rash"),
    ("joint_pain", "Joint pain"),
    ("shortness_of_breath", "Shortness of breath"),
    ("sweating", "Sweating"),
    ("frequent_urination", "Frequent urination"),
    ("excessive_thirst", "Excessive thirst"),
    ("blurred_vision", "Blurred vision"),
    ("night_sweats", "Night sweats"),
    ("weight_loss", "Weight loss"),
    ("stiff_neck", "Stiff neck"),
    ("light_sensitivity", "Light sensitivity"),
]

MODULE_META = {
    "KnowledgeBase": {
        "label": "Knowledge Base & Logic",
        "tag": "First-Order Logic",
        "kind": "diagnostic",
        "description": "Forward/backward chaining over disease rules "
                        "mined from the CSV datasets, with certainty "
                        "factors.",
    },
    "BayesianNet": {
        "label": "Bayesian Network",
        "tag": "Probabilistic Reasoning",
        "kind": "diagnostic",
        "description": "Prior/likelihood table producing a posterior "
                        "distribution over diseases given symptoms.",
    },
    "MLClassifier": {
        "label": "ML Ensemble Classifier",
        "tag": "Decision Tree / RF / GBM",
        "kind": "diagnostic",
        "description": "Decision Tree, Random Forest and Gradient "
                        "Boosting models; the best cross-validated "
                        "model is used at inference time.",
    },
    "NeuralNetwork": {
        "label": "Deep Neural Network",
        "tag": "Dense / BatchNorm / Dropout",
        "kind": "diagnostic",
        "description": "Feed-forward network (128-64-32) trained on "
                        "the same symptom vectors as the ML ensemble.",
    },
    "Fuzzy": {
        "label": "Fuzzy Severity Assessor",
        "tag": "Fuzzy Logic",
        "kind": "support",
        "description": "Fuzzifies temperature, heart rate and symptom "
                        "count into a 0-100 severity score.",
    },
    "Planner": {
        "label": "Treatment Planner",
        "tag": "STRIPS Planning",
        "kind": "support",
        "description": "BFS search over a STRIPS action library to "
                        "build a step-by-step treatment plan.",
    },
}

PEAS = {
    "performance": "Diagnostic accuracy, patient safety, recommendation "
                   "quality, response time",
    "environment": "Hospital / clinic, patient data, EMR",
    "actuators": "Diagnosis report, treatment plan, referral "
                 "recommendation, alerts",
    "sensors": "Symptom input, vitals, lab results, patient history",
    "agent_type": "Model-Based + Goal-Based + Learning",
}

_lock = threading.Lock()

print("=" * 60)
print("Booting diagnostic engine (training ML + neural models)...")
AGENT = build_system()
AGENT._modules["MLClassifier"].train(verbose=False)
AGENT._modules["NeuralNetwork"].train(epochs=10, verbose=0)
print("Engine ready.")
print("=" * 60)


def _new_patient_id() -> str:
    return f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def _serialize_patient(p: PatientPercept) -> dict:
    return {
        "patient_id": p.patient_id,
        "name": p.name,
        "gender": p.gender,
        "symptoms": p.symptoms,
        "age": p.age,
        "temperature": p.temperature,
        "heart_rate": p.heart_rate,
        "blood_pressure": p.blood_pressure,
        "expected_diagnosis": p.expected_diagnosis,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def meta():
    return jsonify({
        "symptoms": [{"id": k, "label": v} for k, v in SYMPTOM_CATALOG],
        "modules": MODULE_META,
        "peas": PEAS,
    })


@app.get("/api/demo-patients")
def demo_patients():
    patients = load_demo_patients(limit=5)
    return jsonify([_serialize_patient(p) for p in patients])


@app.post("/api/diagnose")
def diagnose():
    data = request.get_json(force=True, silent=True) or {}

    checked = data.get("symptoms") or []
    extra = parse_symptoms_input(data.get("extra_symptoms", ""))
    all_symptoms = sorted({s.lower().replace(" ", "_")
                           for s in [*checked, *extra] if s})

    if not all_symptoms:
        return jsonify({"error": "Select at least one symptom."}), 400

    try:
        age = int(data.get("age", 35))
        temperature = float(data.get("temperature", 37.0))
        heart_rate = int(data.get("heart_rate", 80))
    except (TypeError, ValueError):
        return jsonify({"error": "Age, temperature and heart rate must "
                                  "be numbers."}), 400

    expected = data.get("expected_diagnosis")
    percept = PatientPercept(
        patient_id=data.get("patient_id") or _new_patient_id(),
        name=(data.get("name") or "Unknown").strip() or "Unknown",
        gender=(data.get("gender") or "Unknown").strip() or "Unknown",
        symptoms=all_symptoms,
        age=age,
        temperature=temperature,
        heart_rate=heart_rate,
        blood_pressure=(data.get("blood_pressure") or "120/80").strip(),
        expected_diagnosis=normalize_diagnosis(expected) if expected
        else None,
    )

    try:
        with _lock:
            report = AGENT.run(percept)
            report["action_log"] = list(AGENT.memory.action_log[-14:])
            report["performance"] = AGENT.get_performance()
    except Exception as exc:  # pragma: no cover - defensive
        traceback.print_exc()
        return jsonify({"error": f"The agent failed to complete a "
                                  f"diagnosis: {exc}"}), 500

    return jsonify(report)


@app.get("/api/evaluation")
def evaluation():
    metrics_path = REPORT_DIR / "final_system_report.json"
    if not metrics_path.exists():
        return jsonify({"available": False})

    with open(metrics_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    images = {}
    for name, folder in (
        ("confusion_matrix", REPORT_DIR),
        ("module_comparison", REPORT_DIR),
        ("ml_evaluation", IMAGES_DIR),
        ("nn_training", IMAGES_DIR),
    ):
        candidate = folder / f"{name}.png"
        if candidate.exists():
            kind = "reports" if folder == REPORT_DIR else "images"
            images[name] = f"/{kind}/{name}.png"

    return jsonify({
        "available": True,
        "metrics": payload.get("metrics", {}),
        "patients_evaluated": len(payload.get("reports", [])),
        "images": images,
    })


@app.post("/api/evaluation/regenerate")
def regenerate_evaluation():
    try:
        with _lock:
            reports = run_demo(AGENT)
            save_report_outputs(reports)
    except Exception as exc:  # pragma: no cover - defensive
        traceback.print_exc()
        return jsonify({"error": f"Could not regenerate the evaluation "
                                  f"report: {exc}"}), 500
    return evaluation()


@app.get("/reports/<path:filename>")
def report_files(filename):
    return send_from_directory(REPORT_DIR, filename)


@app.get("/images/<path:filename>")
def image_files(filename):
    return send_from_directory(IMAGES_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
