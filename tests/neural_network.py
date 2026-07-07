from modules.neural_network import NeuralDiagnosticModel

def main():
    # Create model
    model = NeuralDiagnosticModel()

    # Train the model (keep epochs small for testing)
    print("Training model...")
    results = model.train(epochs=5, verbose=1)

    print("\nTraining Results")
    print("-" * 30)
    print(results)

    # Test predictions
    test_cases = [
        ["fever", "cough", "fatigue"],
        ["fever", "loss_of_smell", "cough"],
        ["chest_pain", "shortness_of_breath", "sweating"],
        ["frequent_urination", "excessive_thirst", "fatigue"],
        ["headache", "stiff_neck", "light_sensitivity"],
    ]

    print("\nPredictions")
    print("=" * 60)

    for symptoms in test_cases:
        prediction = model.predict(symptoms)

        print(f"\nSymptoms: {symptoms}")
        print(f"Diagnosis : {prediction['diagnosis']}")
        print(f"Confidence: {prediction['confidence']:.2%}")

        # Show top 3 probabilities
        top3 = sorted(
            prediction["all_probs"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        print("Top 3 Predictions:")
        for disease, prob in top3:
            print(f"  {disease:<15} {prob:.2%}")

    # Plot training history
    model.plot_training()


if __name__ == "__main__":
    main()