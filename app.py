# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import sys  
import json  
import warnings  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore') 

import os
# Hide oneDNN optimization logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Hide standard informational TensorFlow logs (0 = all logs, 1 = hide INFO, 2 = hide WARNINGS)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import all modules  
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base import MedicalKnowledgeBase  
from modules.bayesian_net   import SimpleBayesianDiagnostics  
from modules.ml_classifier  import MLDiagnosticClassifier  
from modules.neural_network import NeuralDiagnosticModel  
from modules.fuzzy_controller import FuzzySeverityAssessor  
from modules.planner        import TreatmentPlanner  

# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  

def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  

def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  

def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules"""  
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing modules...")  
    agent.register_module('KnowledgeBase', MedicalKnowledgeBase())  
    agent.register_module('BayesianNet',   SimpleBayesianDiagnostics())  
    agent.register_module('MLClassifier',  MLDiagnosticClassifier())  
    agent.register_module('NeuralNetwork', NeuralDiagnosticModel())  
    agent.register_module('Fuzzy',         FuzzySeverityAssessor())  
    agent.register_module('Planner',       TreatmentPlanner())  

    # Create mock patient data that fits the PatientPercept dataclass requirements
    section("🩺 Simulating Patient Presentation")
    test_patient = PatientPercept(
        patient_id="PAT-2026-001",
        symptoms=["fever", "cough", "shortness_of_breath"],
        age=45,
        temperature=39.2,         # High fever -> will trigger urgent pathways
        heart_rate=105,
        blood_pressure="135/85"
    )
    
    print(f"  Patient ID:  {test_patient.patient_id}")
    print(f"  Symptoms:    {', '.join(test_patient.symptoms)}")
    print(f"  Temperature: {test_patient.temperature}°C")

    # Run the full lifecycle loop automatically via the agent's run method
    section("Running Patient Evaluation Pipeline")
    final_report = agent.run(test_patient)

    # Print the final diagnostic output report nicely formatted
    section("Final Diagnostic Decision Report")
    print(json.dumps(final_report, indent=4))

    # Show the internal workflow history log and metrics
    agent.print_log()
    
    perf = agent.get_performance()
    print(f"\n System Metrics Score: {perf['performance_score']} pts")
    
    return agent

if __name__ == "__main__":
    banner()
    diagnostic_agent = build_system()

 