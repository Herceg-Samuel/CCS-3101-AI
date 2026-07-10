# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks)
# ============================================================

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from modules.data_loader import CSVTrainingDataLoader


class NeuralDiagnosticModel:
    """
    Deep neural network for medical diagnosis.
    Architecture: Input -> Dense -> BN -> Dropout -> Output
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
        self.model = None
        self.history = None
        self.is_trained = False
        self.class_labels = list(self.DISEASE_LABELS)
        self.label_encoder = LabelEncoder()
        self._build_model()

    def _build_model(self, n_outputs: int | None = None):
        n_inputs = len(self.SYMPTOM_FEATURES)
        n_outputs = len(self.class_labels) if n_outputs is None else n_outputs

        self.model = models.Sequential([
            layers.Input(shape=(n_inputs,)),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(n_outputs, activation='softmax'),
        ], name='MedicalDNN')

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

    def _load_training_data(self):
        loader = CSVTrainingDataLoader()
        df = loader.load_training_frame()
        if df.empty:
            raise ValueError(
                'No training rows could be created from the CSV files.')
        return df

    def train(self, epochs: int = 10, verbose: int = 1) -> Dict:
        df = self._load_training_data()
        y = self.label_encoder.fit_transform(df['label'].astype(str))
        self.class_labels = self.label_encoder.classes_.tolist()
        self._build_model(n_outputs=len(self.class_labels))

        X = df[self.SYMPTOM_FEATURES].astype(np.float32).values
        y = y.astype(np.int32)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10,
                restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-6),
        ]

        if verbose:
            print("=" * 55)
            print("  Neural Network - Medical Diagnosis Training")
            print(f"  Architecture: {len(self.SYMPTOM_FEATURES)} -> "
                  f"128 -> 64 -> 32 -> {len(self.class_labels)}")
            print("=" * 55)
            self.model.summary()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=cb_list,
            verbose=verbose,
        )

        val_acc = max(self.history.history['val_accuracy'])
        self.is_trained = True
        if verbose:
            print(f"\nBest Validation Accuracy: {val_acc:.4f}")
        return {'val_accuracy': float(val_acc)}

    def predict(self, symptoms: List[str]) -> Dict:
        if not self.is_trained:
            self.train(verbose=0)

        symptoms_clean = {s.lower().replace(' ', '_') for s in symptoms}
        features = np.array([
            [1.0 if feat in symptoms_clean else 0.0
             for feat in self.SYMPTOM_FEATURES]
        ], dtype=np.float32)

        proba = self.model.predict(features, verbose=0)[0]
        pred_idx = int(np.argmax(proba))
        diagnosis = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            'diagnosis': diagnosis,
            'confidence': round(float(proba[pred_idx]), 4),
            'all_probs': dict(zip(
                self.class_labels, proba.round(4).tolist())),
        }

    def analyze(self, percept) -> Dict:
        result = self.predict(percept.symptoms)
        result['summary'] = (f"DNN: {result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    def plot_training(self):
        if not self.history:
            print("Train model first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metrics = [('accuracy', 'val_accuracy', 'Accuracy'),
                   ('loss', 'val_loss', 'Loss')]
        colors = [('#3498db', '#e74c3c'), ('#2ecc71', '#e67e22')]

        for ax, (train_m, val_m, title), (tc, vc) in zip(
                axes, metrics, colors):
            ax.plot(self.history.history[train_m],
                    color=tc, linewidth=2, label='Train')
            ax.plot(self.history.history[val_m],
                    color=vc, linewidth=2, linestyle='--',
                    label='Validation')
            ax.set_title(f"Model {title}", fontsize=13,
                         fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle("Neural Network Training Curves",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("nn_training.png", dpi=150)
        plt.close()
