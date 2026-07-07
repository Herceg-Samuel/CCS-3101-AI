from modules.knowledge_base import MedicalKnowledgeBase

def print_analysis(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Summary    : {result['summary']}")
    print(f"Diagnosis  : {result['diagnosis']}")
    print(f"Confidence : {result['confidence']:.2%}")

    print("\nInferred Facts")
    print("-" * 40)

    if result["all_inferred"]:
        for fact, cf in sorted(result["all_inferred"].items()):
            print(f"{fact:<30} CF = {cf:.2f}")
    else:
        print("No conclusions inferred.")


def main():

    kb = MedicalKnowledgeBase()

    # Test 1 - Flu

    class FluPatient:
        symptoms = ["fever", "cough", "fatigue"]
        temperature = 38.7
        heart_rate = 90

    result = kb.analyze(FluPatient())
    print_analysis("FLU TEST", result)

    print("\nExplanation")
    print(kb.get_explanation(result["diagnosis"]))

    # Test 2 - COVID

    class CovidPatient:
        symptoms = [
            "fever",
            "cough",
            "fatigue",
            "loss of smell"
        ]
        temperature = 39.0
        heart_rate = 95

    result = kb.analyze(CovidPatient())
    print_analysis("COVID TEST", result)

    print("\nExplanation")
    print(kb.get_explanation(result["diagnosis"]))

    # Test 3 - Cardiac Event

    class CardiacPatient:
        symptoms = [
            "chest pain",
            "shortness of breath",
            "sweating"
        ]
        temperature = 36.9
        heart_rate = 120

    result = kb.analyze(CardiacPatient())
    print_analysis("CARDIAC TEST", result)

    print("\nExplanation")
    print(kb.get_explanation(result["diagnosis"]))

    # Test 4 - Backward Chaining

    print("\n" + "=" * 70)
    print("BACKWARD CHAINING")
    print("=" * 70)

    kb = MedicalKnowledgeBase()

    kb.add_fact("fever")
    kb.add_fact("cough")
    kb.add_fact("fatigue")
    kb.add_fact("high_fever")

    proved, cf = kb.backward_chain("flu_confirmed")

    print(f"Goal       : flu_confirmed")
    print(f"Proved     : {proved}")
    print(f"Confidence : {cf:.2f}")

    # Test 5 - Forward Chaining

    print("\n" + "=" * 70)
    print("FORWARD CHAINING")
    print("=" * 70)

    kb = MedicalKnowledgeBase()

    kb.add_fact("fever")
    kb.add_fact("cough")
    kb.add_fact("fatigue")
    kb.add_fact("high_fever")

    inferred = kb.forward_chain(verbose=True)

    print("\nFinal Inferred Facts")
    print("-" * 40)

    for fact, cf in inferred.items():
        print(f"{fact:<30} CF = {cf:.2f}")

    # Test 6 - Explanation

    print("\n" + "=" * 70)
    print("RULE EXPLANATIONS")
    print("=" * 70)

    diagnoses = [
        "flu_suspected",
        "covid19_suspected",
        "cardiac_event_suspected",
        "meningitis_suspected"
    ]

    for diagnosis in diagnoses:
        print("\n" + diagnosis)
        print(kb.get_explanation(diagnosis))


if __name__ == "__main__":
    main()