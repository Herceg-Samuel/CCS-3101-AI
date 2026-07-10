# ============================================================
# MODULE 1: Intelligent Agent - Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

from dataclasses import dataclass, field
import datetime
from enum import Enum
import re
from typing import Dict, List, Optional


class AgentState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting_symptoms"
    DIAGNOSING = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING = "planning_treatment"
    DONE = "done"


@dataclass
class PatientPercept:
    """What the agent perceives from the environment."""

    patient_id: str
    symptoms: List[str]
    age: int
    temperature: float
    heart_rate: int
    blood_pressure: str
    name: str | None = None
    gender: str | None = None
    expected_diagnosis: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat())


@dataclass
class AgentMemory:
    """Internal model - makes this a model-based agent."""

    patient_history: List[Dict] = field(default_factory=list)
    current_patient: Optional[PatientPercept] = None
    diagnosis_history: List[Dict] = field(default_factory=list)
    action_log: List[str] = field(default_factory=list)


class HealthcareDiagnosticAgent:
    """
    PEAS Definition:
    Performance : Diagnostic accuracy, patient safety, recommendation quality,
                  response time
    Environment : Hospital/clinic, patient data, EMR
    Actuators   : Diagnosis report, treatment plan, referral recommendation,
                  alerts
    Sensors     : Symptom input, vitals, lab results, patient history

    Agent Type  : Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.performance_score = 0
        self._modules = {}
        self._support_modules = {}

    def register_module(self, name: str, module, role: str = "diagnostic"):
        """Plug in AI sub-modules."""
        if role == "diagnostic":
            self._modules[name] = module
        else:
            self._support_modules[name] = module
        print(f"  Module registered: [{name}] ({role})")

    def perceive(self, percept: PatientPercept):
        """Step 1: Perceive the environment."""
        self.memory.current_patient = percept
        self.memory.patient_history.append({
            "id": percept.patient_id,
            "name": percept.name,
            "gender": percept.gender,
            "age": percept.age,
            "symptoms": percept.symptoms,
            "expected_diagnosis": percept.expected_diagnosis,
            "time": percept.timestamp,
        })
        self.state = AgentState.COLLECTING
        self._log(f"Perceived patient {percept.patient_id} "
                  f"with {len(percept.symptoms)} symptoms")
        return self

    def think(self):
        """Step 2: Process and reason."""
        self.state = AgentState.DIAGNOSING
        self._log("Agent thinking: running diagnostic modules...")

        results = {}
        for module_name, module in self._modules.items():
            if hasattr(module, "analyze"):
                result = module.analyze(self.memory.current_patient)
                results[module_name] = result
                self._log(
                    f"  [{module_name}] -> {result.get('summary', 'done')}")

        for module_name, module in self._support_modules.items():
            if module_name.lower() == "planner":
                continue
            if hasattr(module, "analyze"):
                result = module.analyze(self.memory.current_patient)
                results[module_name] = result
                self._log(
                    f"  [{module_name}] -> {result.get('summary', 'done')}")

        self.memory.diagnosis_history.append(results)
        self.state = AgentState.RECOMMENDING
        return results

    def act(self, diagnosis_results: Dict) -> Dict:
        """Step 3: Generate action/recommendation."""
        self.state = AgentState.PLANNING
        patient = self.memory.current_patient

        diagnostic_results = {
            key: value
            for key, value in diagnosis_results.items()
            if key in self._modules and isinstance(value, dict)
        }
        confidences = [
            value.get("confidence", 0)
            for value in diagnostic_results.values()
            if "confidence" in value
        ]
        avg_confidence = sum(confidences) / \
            len(confidences) if confidences else 0.5

        severity = diagnosis_results.get("Fuzzy", {})
        final_diagnosis = self._aggregate_diagnosis(diagnostic_results)
        urgency = self._assess_urgency(patient, avg_confidence, severity)
        treatment_plan = self._build_treatment_plan(final_diagnosis, urgency)

        action_report = {
            "patient_id": patient.patient_id,
            "timestamp": patient.timestamp,
            "symptoms": patient.symptoms,
            "expected_diagnosis": patient.expected_diagnosis,
            "diagnosis": final_diagnosis,
            "confidence": round(avg_confidence, 3),
            "urgency": urgency,
            "severity": severity,
            "module_results": diagnosis_results,
            "recommendations": self._generate_recommendations(urgency),
            "next_action": self._decide_next_action(urgency),
            "treatment_plan": treatment_plan,
        }

        self.performance_score += 10 if avg_confidence > 0.7 else 5
        self.state = AgentState.DONE
        self._log(f"Action generated: {urgency} urgency")
        return action_report

    def run(self, percept: PatientPercept) -> Dict:
        """Full agent cycle: Perceive -> Think -> Act."""
        self.perceive(percept)
        results = self.think()
        return self.act(results)

    def _assess_urgency(self, patient, confidence, severity=None):
        severity_label = ""
        if isinstance(severity, dict):
            severity_label = severity.get("severity_label", "")
        if severity_label in {"CRITICAL", "HIGH"}:
            return severity_label
        if patient.temperature > 39.5 or patient.heart_rate > 120:
            return "CRITICAL"
        if patient.temperature > 38.5 or confidence > 0.8:
            return "HIGH"
        if patient.temperature > 37.5:
            return "MEDIUM"
        return "LOW"

    def _aggregate_diagnosis(self, results):
        weighted = {}
        for value in results.values():
            if not isinstance(value, dict) or "diagnosis" not in value:
                continue
            diagnosis = self._normalize_diagnosis(value.get("diagnosis"))
            if diagnosis == "unknown":
                continue
            weighted[diagnosis] = weighted.get(diagnosis, 0.0) + float(
                value.get("confidence", 0.5))

        if not weighted:
            return "Insufficient data"
        return max(weighted.items(), key=lambda item: item[1])[0]

    def _normalize_diagnosis(self, diagnosis):
        text = str(diagnosis or "unknown").strip().lower()
        text = re.sub(r"(_suspected|_confirmed)$", "", text)
        text = text.replace("-", " ").replace("_", " ")
        aliases = {
            "influenza": "flu",
            "flu": "flu",
            "covid": "covid19",
            "covid 19": "covid19",
            "coronavirus": "covid19",
            "heart disease": "cardiac_event",
            "cardiac": "cardiac_event",
            "cardiac event": "cardiac_event",
            "myocardial infarction": "cardiac_event",
            "common cold": "common_cold",
            "cold": "common_cold",
            "diabetes": "diabetes",
            "diabetes mellitus": "diabetes",
            "dengue": "dengue",
            "meningitis": "meningitis",
            "tuberculosis": "tuberculosis",
            "tb": "tuberculosis",
            "unknown": "unknown",
        }
        return aliases.get(text, text.replace(" ", "_"))

    def _build_treatment_plan(self, diagnosis, urgency):
        planner = self._support_modules.get("Planner")
        if planner and hasattr(planner, "create_treatment_plan"):
            return planner.create_treatment_plan(diagnosis, urgency)
        return {"error": "Planner module not registered", "plan": []}

    def _generate_recommendations(self, urgency):
        base = {
            "CRITICAL": [
                "Immediate emergency consultation required",
                "Alert attending physician now",
                "Transfer to emergency ward",
                "Administer first-line medications",
            ],
            "HIGH": [
                "Schedule urgent appointment within 24 hours",
                "Order blood panel and cultures",
                "Prescribe symptomatic relief",
                "Monitor vitals every 2 hours",
            ],
            "MEDIUM": [
                "Schedule appointment within 3 days",
                "Over-the-counter treatment advised",
                "Monitor temperature twice daily",
                "Increase fluid intake",
            ],
            "LOW": [
                "Home rest recommended",
                "Stay hydrated",
                "Follow up if symptoms worsen",
                "General wellness monitoring",
            ],
        }
        return base.get(urgency, base["LOW"])

    def _decide_next_action(self, urgency):
        actions = {
            "CRITICAL": "EMERGENCY_REFERRAL",
            "HIGH": "URGENT_APPOINTMENT",
            "MEDIUM": "SCHEDULE_FOLLOWUP",
            "LOW": "MONITOR_AT_HOME",
        }
        return actions.get(urgency, "MONITOR_AT_HOME")

    def _log(self, message):
        entry = f"[{self.state.value}] {message}"
        self.memory.action_log.append(entry)

    def print_log(self):
        print("\nAgent Action Log:")
        print("-" * 50)
        for entry in self.memory.action_log:
            print(f"  {entry}")

    def get_performance(self):
        return {
            "total_patients": len(self.memory.patient_history),
            "performance_score": self.performance_score,
            "diagnoses_made": len(self.memory.diagnosis_history),
        }
