from modules.planner import TreatmentPlanner

def print_plan(title, result):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    if "error" in result:
        print("Error:", result["error"])
        return

    print(f"Diagnosis : {result['diagnosis']}")
    print(f"Urgency   : {result['urgency']}")
    print(f"Steps     : {result['steps']}")
    print(f"Duration  : {result['total_duration']}")

    print("\nTreatment Plan")
    print("-" * 60)

    for step in result["plan"]:
        print(
            f"{step['step']}. {step['action']:<25}"
            f" Duration: {step['duration']:<12}"
            f" Cost: {step['cost']}"
        )


def main():
    planner = TreatmentPlanner()

    # Test cases
    test_cases = [
        ("flu", "MEDIUM"),
        ("covid19", "HIGH"),
        ("cardiac_event", "CRITICAL"),
        ("tuberculosis", "HIGH"),
        ("common_cold", "LOW"),
    ]

    for diagnosis, urgency in test_cases:
        plan = planner.create_treatment_plan(diagnosis, urgency)
        print_plan(f"{diagnosis.upper()} ({urgency})", plan)

    # Test analyze() interface (used by HealthcareDiagnosticAgent)
    print("\n" + "=" * 60)
    print("Testing analyze() interface")
    print("=" * 60)

    class DummyPercept:
        symptoms = ["fever", "cough"]

    result = planner.analyze(DummyPercept())

    print(f"Summary    : {result['summary']}")
    print(f"Diagnosis  : {result['diagnosis']}")
    print(f"Confidence : {result['confidence']}")


if __name__ == "__main__":
    main()