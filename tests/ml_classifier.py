from modules.ml_classifier import MLDiagnosticClassifier

def print_prediction(symptoms, result):
    print("\n" + "=" * 70)
    print(f"Symptoms: {', '.join(symptoms)}")
    print("=" * 70)

    print(f"Predicted Disease : {result['diagnosis']}")
    print(f"Confidence        : {result['confidence']:.2%}")
    print(f"Model Used        : {result['model_used']}")

    print("\nTop 5 Predictions")
    print("-" * 40)
    for disease, prob in result["top5"]:
        print(f"{disease:<20} {prob:.2%}")


def main():
    classifier = MLDiagnosticClassifier()

    # Train all models
    results = classifier.train()

    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)

    for model, metrics in results.items():
        print(f"\n{model}")
        print(f"CV Accuracy   : {metrics['cv_mean']:.4f}")
        print(f"CV Std Dev    : {metrics['cv_std']:.4f}")
        print(f"Test Accuracy : {metrics['test_acc']:.4f}")

    print("\nBest Model:", classifier.best_model_name)

    # -------------------------------------------------
    # Test Predictions
    # -------------------------------------------------

    test_cases = [
        ["fever", "cough", "fatigue", "body_aches"],
        ["fever", "loss_of_smell", "cough", "fatigue"],
        ["chest_pain", "shortness_of_breath", "sweating"],
        ["frequent_urination", "excessive_thirst", "blurred_vision"],
        ["headache", "stiff_neck", "light_sensitivity"],
        ["rash", "joint_pain", "fever"],
        ["cough", "night_sweats", "weight_loss"],
    ]

    for symptoms in test_cases:
        result = classifier.predict(symptoms)
        print_prediction(symptoms, result)

    # -------------------------------------------------
    # Test analyze() interface
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("Testing analyze() Interface")
    print("=" * 70)

    class DummyPercept:
        symptoms = [
            "fever",
            "loss_of_smell",
            "fatigue",
            "cough"
        ]

    result = classifier.analyze(DummyPercept())

    print(f"Summary    : {result['summary']}")
    print(f"Diagnosis  : {result['diagnosis']}")
    print(f"Confidence : {result['confidence']:.2%}")

    # -------------------------------------------------
    # Plot Evaluation
    # -------------------------------------------------

    print("\nGenerating evaluation plots...")
    classifier.plot_evaluation()


if __name__ == "__main__":
    main()