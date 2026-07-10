# ============================================================
# MODULE 4: ML Classifier — Supervised Diagnosis
# Covers: Week 9 (Supervised Learning & Decision Trees)
# ============================================================

from modules.data_loader import CSVTrainingDataLoader
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MLDiagnosticClassifier:
    """
    Ensemble ML-based diagnostic classifier.
    Uses Decision Trees, Random Forest, and
    Gradient Boosting for robust diagnosis.
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache',
        'body_aches', 'loss_of_smell', 'chest_pain',
        'rash', 'joint_pain', 'shortness_of_breath',
        'sweating', 'frequent_urination', 'excessive_thirst',
        'blurred_vision', 'night_sweats', 'weight_loss',
        'stiff_neck', 'light_sensitivity'
    ]

    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
    ]

    def __init__(self):
        self.models = {
            'Decision Tree':     DecisionTreeClassifier(
                max_depth=8, criterion='entropy', random_state=42),
            'Random Forest':     RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42),
        }
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def _load_training_data(self) -> pd.DataFrame:
        """Load the bundled CSV files as the training dataset."""
        loader = CSVTrainingDataLoader()
        df = loader.load_training_frame()
        if df.empty:
            raise ValueError(
                'No training rows could be created from the CSV files.')
        return df

    def train(self, verbose: bool = True) -> Dict:
        """Train all models and select the best one"""
        df = self._load_training_data()
        X = df[self.SYMPTOM_FEATURES].values
        y = self.label_encoder.fit_transform(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        results = {}
        best_acc = 0.0

        if verbose:
            print("=" * 55)
            print("  ML Diagnostic Classifier — Training")
            print("=" * 55)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X, y, cv=5)
            results[name] = {
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'test_acc': float(test_acc)
            }
            if verbose:
                print(f"\n  🌲 {name}")
                print(f"     Test Accuracy: {test_acc:.4f}")

            if cv_scores.mean() > best_acc:
                best_acc = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name

        self.is_trained = True
        self._X_test = X_test
        self._y_test = y_test

        if verbose:
            print(f"\n  🏆 Best Model: {self.best_model_name} "
                  f"({best_acc:.4f})")
        return results

    def predict(self, symptoms: List[str]) -> Dict:
        """Predict disease from symptom list"""
        if not self.is_trained:
            self.train(verbose=False)

        symptoms_clean = {s.lower().replace(' ', '_') for s in symptoms}
        features = np.array([
            [1 if s in symptoms_clean else 0
             for s in self.SYMPTOM_FEATURES]
        ])
        pred_encoded = self.best_model.predict(features)[0]
        pred_proba = self.best_model.predict_proba(features)[0]

        disease = self.label_encoder.inverse_transform([pred_encoded])[0]
        classes = self.label_encoder.inverse_transform(
            range(len(pred_proba)))
        prob_map = dict(zip(classes, pred_proba))
        top5 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'diagnosis':      disease,
            'confidence':     round(float(pred_proba[pred_encoded]), 4),
            'top5':           top5,
            'model_used':     self.best_model_name,
            'symptom_vector': features[0].tolist()
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
        result['summary'] = (f"{result['model_used']}: "
                             f"{result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    def plot_evaluation(self):
        """Visualize model performance"""
        if not self.is_trained:
            self.train(verbose=False)

        y_pred = self.best_model.predict(self._X_test)
        cm = confusion_matrix(self._y_test, y_pred)
        labels = self.label_encoder.classes_

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title(f"Confusion Matrix\n({self.best_model_name})",
                          fontweight='bold')
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Feature Importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:12]
            top_features = [self.SYMPTOM_FEATURES[i] for i in sorted_idx]
            top_values = importances[sorted_idx]
            colors = plt.cm.RdYlGn(top_values / top_values.max())
            axes[1].barh(range(len(top_features)), top_values[::-1],
                         color=colors[::-1])
            axes[1].set_yticks(range(len(top_features)))
            axes[1].set_yticklabels(top_features[::-1])
            axes[1].set_title("Feature Importances (Top 12)",
                              fontweight='bold')
            axes[1].set_xlabel("Importance Score")

        plt.suptitle(f"ML Diagnostic Model Evaluation — {self.best_model_name}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("ml_evaluation.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Saved: ml_evaluation.png")
