from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class CSVTrainingDataLoader:
    """Build training rows from the bundled CSV data files."""

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache',
        'body_aches', 'loss_of_smell', 'chest_pain',
        'rash', 'joint_pain', 'shortness_of_breath',
        'sweating', 'frequent_urination', 'excessive_thirst',
        'blurred_vision', 'night_sweats', 'weight_loss',
        'stiff_neck', 'light_sensitivity'
    ]

    LABEL_ALIASES = {
        'influenza': 'flu',
        'flu': 'flu',
        'covid': 'covid19',
        'covid19': 'covid19',
        'covid 19': 'covid19',
        'common cold': 'common_cold',
        'diabetes': 'diabetes',
        'heart disease': 'cardiac_event',
        'cardiac event': 'cardiac_event',
        'dengue': 'dengue',
        'tuberculosis': 'tuberculosis',
        'meningitis': 'meningitis',
    }

    TARGET_LABELS = {
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
    }

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path(
            __file__).resolve().parent.parent / 'data'

    def _normalize(self, value: object) -> str:
        if value is None:
            return ''
        return str(value).strip().lower()

    def _normalize_label(self, value: object) -> str:
        text = self._normalize(value).replace('-', ' ')
        return self.LABEL_ALIASES.get(text, text.replace(' ', '_'))

    def _feature_vector_from_text(self, text: str) -> Dict[str, int]:
        features = {feature: 0 for feature in self.SYMPTOM_FEATURES}
        pool = self._normalize(text)

        if 'fever' in pool:
            features['fever'] = 1
        if 'cough' in pool or 'coughing' in pool:
            features['cough'] = 1
        if 'fatigue' in pool or 'tired' in pool or 'weakness' in pool or 'weak' in pool:
            features['fatigue'] = 1
        if 'headache' in pool or 'headaches' in pool:
            features['headache'] = 1
        if 'body ache' in pool or 'body aches' in pool or 'muscle pain' in pool or 'aches' in pool:
            features['body_aches'] = 1
        if 'loss of smell' in pool or 'anosmia' in pool or 'smell' in pool:
            features['loss_of_smell'] = 1
        if 'chest pain' in pool or 'chest' in pool:
            features['chest_pain'] = 1
        if 'rash' in pool:
            features['rash'] = 1
        if 'joint pain' in pool or 'joint' in pool or 'arthralgia' in pool:
            features['joint_pain'] = 1
        if 'shortness of breath' in pool or 'difficulty breathing' in pool or 'breath' in pool or 'breathing' in pool:
            features['shortness_of_breath'] = 1
        if 'sweating' in pool or 'sweat' in pool:
            features['sweating'] = 1
        if 'frequent urination' in pool or 'urination' in pool or 'urinate' in pool:
            features['frequent_urination'] = 1
        if 'excessive thirst' in pool or 'thirst' in pool:
            features['excessive_thirst'] = 1
        if 'blurred vision' in pool or 'vision' in pool:
            features['blurred_vision'] = 1
        if 'night sweats' in pool or 'night sweat' in pool:
            features['night_sweats'] = 1
        if 'weight loss' in pool or 'weight' in pool:
            features['weight_loss'] = 1
        if 'stiff neck' in pool or 'neck' in pool:
            features['stiff_neck'] = 1
        if 'light sensitivity' in pool or 'photosensitivity' in pool or 'light' in pool:
            features['light_sensitivity'] = 1
        return features

    def _yes(self, value: object) -> bool:
        return self._normalize(value) == 'yes'

    def _feature_vector_from_disease_row(self, row) -> Dict[str, int]:
        features = {feature: 0 for feature in self.SYMPTOM_FEATURES}
        if self._yes(row.get('Fever')):
            features['fever'] = 1
        if self._yes(row.get('Cough')):
            features['cough'] = 1
        if self._yes(row.get('Fatigue')):
            features['fatigue'] = 1
        if self._yes(row.get('Difficulty Breathing')):
            features['shortness_of_breath'] = 1

        label = self._normalize_label(row.get('Disease'))
        hints = {
            'flu': ['body_aches', 'headache'],
            'covid19': ['loss_of_smell'],
            'dengue': ['rash', 'joint_pain', 'headache'],
            'cardiac_event': ['chest_pain', 'sweating'],
            'diabetes': ['frequent_urination', 'excessive_thirst',
                         'blurred_vision'],
            'tuberculosis': ['night_sweats', 'weight_loss'],
            'meningitis': ['stiff_neck', 'light_sensitivity', 'high_fever'],
            'common_cold': ['headache'],
        }
        for feature in hints.get(label, []):
            if feature in features:
                features[feature] = 1
        return features

    def _add_synthetic_rows(self, rows: List[Dict[str, object]],
                            per_label: int = 30):
        profiles = {
            'flu': {
                'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                'headache': 0.70, 'body_aches': 0.80,
            },
            'covid19': {
                'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                'loss_of_smell': 0.85, 'shortness_of_breath': 0.45,
            },
            'dengue': {
                'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                'headache': 0.90, 'body_aches': 0.88,
            },
            'cardiac_event': {
                'chest_pain': 0.92, 'shortness_of_breath': 0.88,
                'sweating': 0.75, 'fatigue': 0.70,
            },
            'diabetes': {
                'frequent_urination': 0.95, 'excessive_thirst': 0.92,
                'blurred_vision': 0.70, 'fatigue': 0.82,
            },
            'common_cold': {
                'cough': 0.90, 'fever': 0.50, 'headache': 0.60,
                'fatigue': 0.55,
            },
            'tuberculosis': {
                'cough': 0.90, 'night_sweats': 0.85,
                'weight_loss': 0.82, 'fatigue': 0.80, 'fever': 0.65,
            },
            'meningitis': {
                'headache': 0.95, 'stiff_neck': 0.90,
                'light_sensitivity': 0.85, 'fever': 0.88,
            },
        }
        existing = {}
        for row in rows:
            existing[row['label']] = existing.get(row['label'], 0) + 1

        rng = np.random.default_rng(42)
        for label, profile in profiles.items():
            needed = max(0, per_label - existing.get(label, 0))
            for _ in range(needed):
                features = {feature: 0 for feature in self.SYMPTOM_FEATURES}
                for feature, probability in profile.items():
                    features[feature] = int(rng.random() < probability)
                rows.append({'label': label, **features})

    def load_training_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []

        disease_file = self.data_dir / 'diseases.csv'
        symptoms_file = self.data_dir / 'symptoms.csv'
        patient_file = self.data_dir / 'patient_records.csv'

        if disease_file.exists():
            disease_df = pd.read_csv(disease_file)
            for _, row in disease_df.iterrows():
                label = self._normalize_label(row.get('Disease')) or self._normalize_label(
                    row.get('Condition'))
                if not label or label not in self.TARGET_LABELS:
                    continue
                features = self._feature_vector_from_disease_row(row)
                rows.append({'label': label, **features})

        if symptoms_file.exists():
            symptom_df = pd.read_csv(symptoms_file)
            for _, row in symptom_df.iterrows():
                label = self._normalize_label(
                    row.get('Name')) or self._normalize_label(row.get('Disease'))
                if not label or label not in self.TARGET_LABELS:
                    continue
                text = ' '.join([
                    self._normalize(row.get('Name')),
                    self._normalize(row.get('Symptoms')),
                    self._normalize(row.get('Treatments')),
                ])
                features = self._feature_vector_from_text(text)
                rows.append({'label': label, **features})

        if patient_file.exists():
            patient_df = pd.read_csv(patient_file)
            for _, row in patient_df.iterrows():
                label = self._normalize_label(
                    row.get('Condition')) or self._normalize_label(row.get('Outcome'))
                if not label or label not in self.TARGET_LABELS:
                    continue
                text = ' '.join([
                    self._normalize(row.get('Condition')),
                    self._normalize(row.get('Procedure')),
                    self._normalize(row.get('Outcome')),
                ])
                features = self._feature_vector_from_text(text)
                rows.append({'label': label, **features})

        if not rows:
            return pd.DataFrame(columns=['label', *self.SYMPTOM_FEATURES])

        self._add_synthetic_rows(rows)
        return pd.DataFrame(rows)
