from modules.bayesian_net import SimpleBayesianDiagnostics

def print_prediction(symptoms, result):
    print("\n" + "=" * 70)
    print(f"Symptoms: {', '.join(symptoms)}")
    print("=" * 70)

    print(f"Predicted Diagnosis : {result['diagnosis']}")
    print(f"Confidence          : {result['confidence']:.2%}")

    print("\nTop Diagnoses")
    print("-" * 40)
    for disease, prob in result["ranked_diagnoses"]:
        print(f"{disease:<20} {prob:.2%}")


def main():
    model = SimpleBayesianDiagnostics()

    # Test Cases

    test_cases = [
        ["fever", "cough", "fatigue"],
        ["fever", "loss_of_smell", "cough"],
        ["rash", "joint_pain", "fever"],
        ["chest_pain", "shortness_of_breath", "sweating"],
        ["frequent_urination", "excessive_thirst", "blurred_vision"],
        ["headache", "body_aches", "fatigue"],
        [],
    ]

    for symptoms in test_cases:
        result = model.analyze(type("Dummy", (), {"symptoms": symptoms})())

        print_prediction(symptoms if symptoms else ["<none>"], result)

        print("\nExplanation")
        print("-" * 40)
        print(model.explain(result["diagnosis"], symptoms))

    # Direct Posterior Test

    print("\n" + "=" * 70)
    print("Posterior Probability Distribution")
    print("=" * 70)

    symptoms = ["fever", "cough", "fatigue"]

    posteriors = model.compute_posterior(symptoms)

    for disease, probability in sorted(
        posteriors.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{disease:<20} {probability:.2%}")

    print(f"\nTotal Probability = {sum(posteriors.values()):.4f}")


if __name__ == "__main__":
    main()