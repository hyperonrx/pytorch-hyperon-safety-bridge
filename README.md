# NeuroIslet: A Neuro-Symbolic Artificial Pancreas 🧠+💉

**NeuroIslet** is a demonstration of **Cognitive Synergy** applied to life-critical medical control. It combines the intuitive forecasting power of Deep Learning with the verifiable safety of Symbolic Logic.

## 🚀 The Architecture

NeuroIslet implements a **System 1 + System 2** architecture using the [OpenCog Hyperon](https://github.com/trueagi-io/hyperon-experimental) framework.

### 1. System 1: The Intuition (PyTorch)
* **Model:** Temporal Fusion Transformer (TFT).
* **Role:** Forecasts future glucose trajectories based on continuous sensor data (CGM, Insulin, Carbs).
* **Output:** Probabilistic forecasts with uncertainty quantiles.

### 2. The Bridge: Quantile Symbol Grounding
* **The Innovation:** We do not attempt to ground every raw data point. Instead, we extract **Semantic Features** from the neural output:
    1.  **Immediate Uncertainty** (q90-q10 width at 30min).
    2.  **Strategic Uncertainty** (q90-q10 width at 60min).
    3.  **Projected Velocity** (Slope of q50).
* These features are converted into **MeTTa Atoms** in real-time.

### 3. System 2: The Guardian (Hyperon)
* **Language:** MeTTa (Meta Type Talk).
* **Role:** Executes deterministic safety rules against the grounded atoms.
* **Capability:** It can **VETO** neural actions if they violate safety axioms, regardless of the neural network's confidence.

## 🧪 Verification: The "Pizza Stress Test"

To validate the architecture, we perform a "Man-in-the-Middle" injection of noise into the simulation.

* **Scenario:** A "Ghost Meal" creates a glucose spike invisible to the neural net's history.
* **Neural Response:** The TFT becomes confused (High Variance) and attempts a reactive "Panic Bolus."
* **Symbolic Response:** The Hyperon layer detects `(State (Uncertainty High))` and executes:
    ```lisp
    (= (safety-check $uncertainty)
       (if (> $uncertainty 50)
           (Block-Action "Neural Confidence Too Low")
           (Allow-Action)))
    ```
* **Result:** The dangerous action is blocked, preventing potential hypoglycemia.

## 🛠️ Usage (Docker)

This project runs inside the standard Hyperon Alpha container.

```bash
# 1. Pull the image
docker pull trueagi/hyperon:latest

# 2. Run the agent
docker run -it -v ${PWD}:/app -w /app trueagi/hyperon:latest python pancreas_hyperon.py
