# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

import math
from collections import defaultdict
from pathlib import Path
from typing import Set, List, Dict, Tuple, Optional

import pandas as pd

from modules.data_loader import CSVTrainingDataLoader


class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.
    Supports forward chaining, backward chaining,
    and confidence-weighted inference.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path(
            __file__).resolve().parent.parent / 'data'
        self.facts:  Set[str] = set()
        self.rules:  List[Tuple] = []
        self.certainty_factors: Dict[str, float] = {}
        self.loader = CSVTrainingDataLoader(self.data_dir)
        self.disease_counts: Dict[str, int] = {}
        self.symptom_probabilities: Dict[str, Dict[str, float]] = {}
        self._load_medical_knowledge()

    def _normalize_label(self, value: object) -> str:
        return self.loader._normalize_label(value)

    def _load_medical_knowledge(self):
        """Load domain medical knowledge from CSV datasets."""
        self._train_from_csv_files()

        # Build symptom-based suspect rules for backward compatibility
        disease_rules = self._build_disease_rules_from_data()
        for conditions, conclusion, cf in disease_rules:
            self.add_rule(conditions, conclusion, cf)

        # Add static confirmation and urgency rules
        confirmation_rules = [
            (["flu_suspected", "high_fever"],
             "flu_confirmed", 0.85),
            (["covid19_suspected", "positive_pcr"],
             "covid19_confirmed", 0.99),
            (["cardiac_event_suspected", "elevated_troponin"],
             "myocardial_infarction", 0.95),
        ]
        urgency_rules = [
            (["myocardial_infarction"], "EMERGENCY", 1.00),
            (["meningitis_suspected"], "EMERGENCY", 0.95),
            (["covid19_confirmed"], "ISOLATE_AND_TREAT", 0.99),
            (["flu_confirmed"], "REST_AND_MEDICATE", 0.90),
        ]
        for conditions, conclusion, cf in confirmation_rules + urgency_rules:
            self.add_rule(conditions, conclusion, cf)

    def _train_from_csv_files(self):
        """Train disease symptom distributions from CSV datasets."""
        disease_counts = defaultdict(int)
        symptom_counts = defaultdict(lambda: defaultdict(int))

        disease_file = self.data_dir / 'diseases.csv'
        symptoms_file = self.data_dir / 'symptoms.csv'
        patient_file = self.data_dir / 'patient_records.csv'

        if disease_file.exists():
            df = pd.read_csv(disease_file)
            for _, row in df.iterrows():
                label = self._normalize_label(row.get('Disease'))
                if not label:
                    continue
                disease_counts[label] += 1
                features = self.loader._feature_vector_from_disease_row(row)
                for feature, value in features.items():
                    if value:
                        symptom_counts[label][feature] += 1

        if symptoms_file.exists():
            df = pd.read_csv(symptoms_file)
            for _, row in df.iterrows():
                label = self._normalize_label(row.get('Name'))
                if not label:
                    continue
                text = ' '.join([
                    self.loader._normalize(row.get('Name')),
                    self.loader._normalize(row.get('Symptoms')),
                    self.loader._normalize(row.get('Treatments')),
                ])
                features = self.loader._feature_vector_from_text(text)
                if any(features.values()):
                    disease_counts[label] += 1
                    for feature, value in features.items():
                        if value:
                            symptom_counts[label][feature] += 1

        if patient_file.exists():
            df = pd.read_csv(patient_file)
            for _, row in df.iterrows():
                label = self._normalize_label(row.get('Condition'))
                if not label:
                    continue
                text = ' '.join([
                    self.loader._normalize(row.get('Condition')),
                    self.loader._normalize(row.get('Procedure')),
                    self.loader._normalize(row.get('Outcome')),
                ])
                features = self.loader._feature_vector_from_text(text)
                if any(features.values()):
                    disease_counts[label] += 1
                    for feature, value in features.items():
                        if value:
                            symptom_counts[label][feature] += 1

        self.disease_counts = {
            disease: count for disease, count in disease_counts.items()}
        total_cases = sum(disease_counts.values()) or 1
        self.symptom_probabilities = {}

        for disease, counts in symptom_counts.items():
            total = disease_counts[disease]
            disease_prior = (total + 1) / (total_cases + len(disease_counts))
            self.symptom_probabilities[disease] = {
                feature: min(0.99, (count + 1) / (total + 2))
                for feature, count in counts.items()
            }
            for feature in self.loader.SYMPTOM_FEATURES:
                self.symptom_probabilities[disease].setdefault(feature, 0.01)
            self.symptom_probabilities[disease]['__prior__'] = disease_prior

    def _build_disease_rules_from_data(self) -> List[Tuple[List[str], str, float]]:
        """Infer simple suspect rules from the trained distributions."""
        inferred_rules = []
        for disease, probs in self.symptom_probabilities.items():
            candidate_symptoms = sorted(
                [feature for feature in probs if feature != '__prior__'],
                key=lambda f: probs[f],
                reverse=True
            )[:4]
            if not candidate_symptoms:
                continue
            confidence = min(0.95, 0.45 + 0.12 * len(candidate_symptoms))
            conclusion = f"{disease}_suspected"
            inferred_rules.append(
                (candidate_symptoms, conclusion, round(confidence, 2)))
        return inferred_rules

    def _score_disease_from_symptoms(self, symptoms: List[str]) -> Dict[str, float]:
        """Compute disease probability scores from patient symptoms."""
        symptoms_clean = [s.lower().replace(' ', '_') for s in symptoms]
        scores = {}
        for disease, probs in self.symptom_probabilities.items():
            prior = probs.get('__prior__', 0.01)
            log_score = math.log(prior)
            for symptom in symptoms_clean:
                p = probs.get(symptom, 0.01)
                log_score += math.log(p)
            scores[disease] = math.exp(log_score)

        total = sum(scores.values()) or 1.0
        return {disease: round(score / total, 4) for disease, score in scores.items()}

    def add_fact(self, fact: str, certainty: float = 1.0):
        self.facts.add(fact)
        self.certainty_factors[fact] = certainty

    def add_rule(self, conditions: List[str],
                 conclusion: str, certainty: float = 1.0):
        self.rules.append((conditions, conclusion, certainty))

    def load_patient_symptoms(self, symptoms: List[str]):
        """Load patient symptoms as facts"""
        for symptom in symptoms:
            self.add_fact(symptom.lower().replace(' ', '_'))

    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """Forward chaining with certainty factors"""
        inferred = {}
        changed = True
        iteration = 0

        while changed:
            changed = False
            iteration += 1
            for conditions, conclusion, rule_cf in self.rules:
                all_known = all(
                    c in self.facts or c in inferred for c in conditions
                )
                if all_known and conclusion not in inferred:
                    # Combine certainty factors
                    cond_cfs = [
                        self.certainty_factors.get(c,
                                                   inferred.get(c, 1.0))
                        for c in conditions
                    ]
                    combined_cf = rule_cf * min(cond_cfs)
                    inferred[conclusion] = round(combined_cf, 4)

                    if verbose:
                        cond_str = " ∧ ".join(conditions)
                        print(f"  Iter {iteration}: "
                              f"{cond_str} → {conclusion} "
                              f"(CF={combined_cf:.3f})")
                    changed = True
        return inferred

    def backward_chain(self, goal: str,
                       visited: Optional[Set] = None,
                       depth: int = 0) -> Tuple[bool, float]:
        """Backward chaining — prove a goal"""
        indent = "  " * depth
        visited = visited or set()

        if goal in self.facts:
            return True, self.certainty_factors.get(goal, 1.0)
        if goal in visited:
            return False, 0.0
        visited.add(goal)

        for conditions, conclusion, rule_cf in self.rules:
            if conclusion == goal:
                results = [
                    self.backward_chain(c, visited.copy(), depth+1)
                    for c in conditions
                ]
                if all(proved for proved, _ in results):
                    cf = rule_cf * min(cf for _, cf in results)
                    return True, round(cf, 4)
        return False, 0.0

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        self.facts = set()
        self.certainty_factors = {}
        self.load_patient_symptoms(percept.symptoms)

        # Add vitals as facts
        if percept.temperature > 38.0:
            self.add_fact("fever",
                          min(1.0, (percept.temperature - 37.0) / 3.0))
        if percept.temperature > 39.5:
            self.add_fact("high_fever", 1.0)
        if percept.heart_rate > 100:
            self.add_fact("tachycardia", 1.0)

        inferred = self.forward_chain()
        diseases = {k: v for k, v in inferred.items()
                    if 'suspected' in k or 'confirmed' in k}

        probability_scores = self._score_disease_from_symptoms(
            percept.symptoms)
        top_csv_disease = max(probability_scores, key=probability_scores.get)
        top_csv_confidence = probability_scores[top_csv_disease]

        if diseases:
            top_rule_disease = max(diseases, key=diseases.get)
            top_rule_confidence = diseases[top_rule_disease]
        else:
            top_rule_disease = None
            top_rule_confidence = 0.0

        diagnosis = top_rule_disease or top_csv_disease
        confidence = top_rule_confidence if top_rule_confidence >= top_csv_confidence else top_csv_confidence
        if top_rule_disease and top_csv_disease and top_rule_disease.split('_')[0] != top_csv_disease:
            diagnosis = top_csv_disease
            confidence = top_csv_confidence

        return {
            'summary':    f"Inferred {len(inferred)} conclusions; best CSV match: {top_csv_disease}",
            'diagnosis':  diagnosis,
            'confidence': round(confidence, 4),
            'all_inferred': inferred,
            'csv_scores': probability_scores,
        }

    def get_explanation(self, diagnosis: str) -> str:
        """Explain how a diagnosis was reached"""
        for conditions, conclusion, cf in self.rules:
            if conclusion == diagnosis:
                return (f"'{diagnosis}' derived from: "
                        f"{' + '.join(conditions)} (CF={cf})")
        return f"'{diagnosis}' is a base fact"
