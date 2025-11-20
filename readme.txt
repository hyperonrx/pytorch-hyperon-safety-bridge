# PyTorch-Hyperon Safety Bridge 🌉
# A Neuro-Symbolic Integration Framework for Safe AI Control Systems

## 📖 Overview

**PyTorch-Hyperon Safety Bridge** is a containerized framework that solves the "Black Box" problem in Deep Reinforcement Learning (DRL). It establishes a bidirectional communication channel between **PyTorch** (Deep Learning/Intuition) and **OpenCog Hyperon** (Symbolic Logic/Reasoning).

This repository contains a reference implementation called **"GlucoVision"**: an Artificial Pancreas control system. It demonstrates how a Symbolic "Guardian Layer" (written in **MeTTa**) can audit, constrain, and veto the actions of a Neural Network agent in a life-critical medical environment.

## 🧠 The "Cognitive Synergy" Architecture

This project implements Ben Goertzel's vision of **Cognitive Synergy** by combining two distinct systems:

1.  **System 1 (Neural Intuition):**
    *   **Engine:** PyTorch (Temporal Fusion Transformer + PPO Agent).
    *   **Role:** Analyzes complex time-series glucose data to predict trends and propose insulin dosages.
    *   **Weakness:** Prone to "hallucinations" (confident but wrong actions) in high-noise scenarios.

2.  **System 2 (Symbolic Reasoning):**
    *   **Engine:** OpenCog Hyperon (MeTTa).
    *   **Role:** Acts as an immutable **Safety Constitution**. It validates every proposed action against physiological rules and uncertainty thresholds.
    *   **Strength:** Transparent, logical, and auditable.

## 🛡️ The "Bridge" Mechanics

The core innovation is the Dockerized runtime that allows Python Tensors to trigger MeTTa atoms.

1.  **Inference:** The RL Agent proposes an Action (e.g., `Bolus 2.0U`) and calculates an **Uncertainty Score**.
2.  **Injection:** The Bridge constructs a dynamic MeTTa expression:
    ```lisp
    !(evaluate-safety <Action_ID> <Glucose_Value> <IOB_Value> <Uncertainty_Score>)
    ```
3.  **Reasoning:** The Hyperon AtomSpace evaluates this against `safety_logic.metta`.
4.  **Verdict:** If the Neural Net is confused (High Uncertainty) or violates safety rules (Low Glucose), Hyperon returns a **Modified Action**.

## ⚠️ Important: Data & Artifacts

**Note regarding Repository Size:**
To maintain a clean and lightweight repository, the massive datasets and pre-trained model checkpoints are **not included** in this Git repo.

*   **Missing Data:** The `subjects/` and `ohio_demo/` folders (patient data) are excluded.
*   **Missing Model:** The pre-trained `tft_forecasting_model.ckpt` (100MB+) is excluded.

**How to Run:**
The code includes a **Fallback Mode**. If the external data or models are missing, the system will:
1.  Use the default Simglucose virtual patient generator.
2.  Run the Symbolic Bridge logic without the TFT forecasting signal (relying on hard logic constraints).

*To reproduce the full experiment with specific patient data, please mount your local data folders to the Docker container at runtime.*

## 🚀 Installation & Usage

This project is fully Dockerized to resolve the complex dependency chains required to run Rust-based Hyperon alongside PyTorch on non-Linux systems.

### 1. Prerequisites
*   Docker Desktop

### 2. Build the Brain
```bash
docker build -t hyperon-pancreas .
3. Run the Simulation
code
Bash
docker run -v %cd%/artifacts_optimization_run:/app/artifacts_optimization_run hyperon-pancreas

📊 Validation: The "Stress Test"
To validate the bridge, we performed a "Man-in-the-Middle" Stress Test at Step 15 of the simulation:
Injection: We forced the Neural Network to hallucinate a dangerous overdose (Large Bolus) and artificially injected a high Uncertainty signal (100.0).
Response: The Hyperon Guardian successfully detected the high uncertainty.

Log Output:
[MeTTa] UNCERTAINTY: Neural confidence low. Aggressive bolus reduced.
Result: The dangerous action was blocked and replaced with a safe fallback action.

🔮 Roadmap
This project acts as the Alpha Prototype for a larger initiative proposed for SingularityNET Deep Funding:
Phase 1 (Complete): Static Logic Bridge (Hard-coded rules).
Phase 2: Dynamic Knowledge Graph – Using patient context to rewrite safety rules in real-time.
Phase 3: Inductive Logic Programming – Enabling the MeTTa layer to learn new rules from "near-miss" events.
Author: Benjamin Nancarrow