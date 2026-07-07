from modules.agent          import HealthcareDiagnosticAgent, PatientPercept 

class DummyDiagnosticModule:
    """Simple mock diagnostic module for testing."""

    def analyze(self, patient):
        symptoms = [s.lower() for s in patient.symptoms]

        if "fever" in symptoms and "cough" in symptoms:
            return {
                "diagnosis": "Flu",
                "confidence": 0.85,
                "summary": "Likely influenza"
            }

        elif "headache" in symptoms:
            return {
                "diagnosis": "Migraine",
                "confidence": 0.75,
                "summary": "Possible migraine"
            }

        return {
            "diagnosis": "Unknown",
            "confidence": 0.50,
            "summary": "Insufficient evidence"
        }


def main():
    # Create the agent
    agent = HealthcareDiagnosticAgent()

    # Register a test diagnostic module
    agent.register_module("Dummy Diagnosis", DummyDiagnosticModule())

    # Create a sample patient
    patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "fatigue"],
        age=30,
        temperature=39.2,
        heart_rate=95,
        blood_pressure="120/80"
    )

    # Run the complete agent cycle
    report = agent.run(patient)

    # Display report
    print("\n=== Diagnostic Report ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    # Display action log
    agent.print_log()

    # Display performance metrics
    print("\n=== Performance Metrics ===")
    print(agent.get_performance())


if __name__ == "__main__":
    main()