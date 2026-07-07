from modules.fuzzy_controller import FuzzySeverityAssessor

def print_result(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Severity Score : {result['severity_score']}/100")
    print(f"Severity Label : {result['severity_label']}")

    print("\nRule Strengths")
    print("-" * 40)
    for rule, strength in result["rule_strengths"].items():
        print(f"{rule:<10}: {strength:.3f}")

    print("\nMembership Values")
    print("-" * 40)

    for category, memberships in result["memberships"].items():
        print(f"\n{category.capitalize()}:")
        for label, value in memberships.items():
            print(f"  {label:<10}: {value:.3f}")


def main():
    assessor = FuzzySeverityAssessor()

    test_cases = [
        {
            "name": "Healthy Patient",
            "temp": 36.8,
            "hr": 76,
            "symptoms": 1
        },
        {
            "name": "Mild Flu",
            "temp": 38.0,
            "hr": 88,
            "symptoms": 3
        },
        {
            "name": "High Fever",
            "temp": 39.2,
            "hr": 110,
            "symptoms": 6
        },
        {
            "name": "Critical Emergency",
            "temp": 40.4,
            "hr": 130,
            "symptoms": 8
        },
        {
            "name": "Borderline Case",
            "temp": 37.6,
            "hr": 95,
            "symptoms": 4
        }
    ]

    for case in test_cases:
        result = assessor.assess(
            case["temp"],
            case["hr"],
            case["symptoms"]
        )

        print_result(case["name"], result)

    # Test the agent interface
    print("\n" + "=" * 70)
    print("Testing analyze() Interface")
    print("=" * 70)

    class DummyPercept:
        temperature = 39.4
        heart_rate = 118
        symptoms = [
            "fever",
            "cough",
            "fatigue",
            "headache",
            "body aches",
            "shortness of breath"
        ]

    result = assessor.analyze(DummyPercept())

    print(f"Summary    : {result['summary']}")
    print(f"Diagnosis  : {result['diagnosis']}")
    print(f"Confidence : {result['confidence']:.2f}")


if __name__ == "__main__":
    main()