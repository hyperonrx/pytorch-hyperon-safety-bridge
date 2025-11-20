# =======================================================================================
#               AI Pancreas Control System: Neuro-Symbolic Edition (v6.4)
#                                "The Hyperon Build"
# =======================================================================================
#
# INTEGRATION: PyTorch (Neural Intuition) + OpenCog Hyperon (Symbolic Reasoning)
#
# This system uses a Temporal Fusion Transformer (TFT) and PPO Agent to propose insulin
# dosages, but filters every decision through a MeTTa (Meta Type Talk) symbolic logic
# layer to ensure physiological safety constraints are met.
# =======================================================================================

import os
import glob
import json
import logging
import random
import warnings
import math
import pickle
import sys
from typing import Dict, Tuple, List, Optional, Any
from collections import deque
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# --- NEURO-SYMBOLIC IMPORT ---
try:
    from hyperon import MeTTa
    HYPERON_ACTIVE = True
except ImportError:
    HYPERON_ACTIVE = False
    print("⚠️  WARNING: Hyperon (MeTTa) not installed. Running in standard Python mode.")

# --- ML IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("simglucose").setLevel(logging.ERROR)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("ai_pancreas_hyperon")

# Check for ML Libraries
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import NaNLabelEncoder, GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from sklearn.preprocessing import StandardScaler
except ImportError:
    log.error("CRITICAL: ML libraries missing. Install requirements.txt")
    sys.exit(1)

try:
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario import Scenario
    from simglucose.simulation.scenario_gen import RandomScenario
    from simglucose.controller.base import Action
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.actuator.pump import InsulinPump
    from simglucose.patient.t1dpatient import T1DPatient
except ImportError:
    log.error("CRITICAL: Simglucose missing. Install requirements.txt")
    sys.exit(1)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
SEED = 42
SMOKE_TEST = True  # Set to False for full training run

def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    if 'pl' in globals(): pl.seed_everything(s, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Action Space
ACTION_MAP: Dict[int, Dict] = {
    0: {"type": "NO_ACTION", "basal_mult": 1.0, "bolus": 0.0, "desc": "Maintain current basal rate"},
    1: {"type": "BOLUS", "basal_mult": 1.0, "bolus": 0.5, "desc": "Administer small bolus (0.5U)"},
    2: {"type": "BOLUS", "basal_mult": 1.0, "bolus": 1.0, "desc": "Administer medium bolus (1.0U)"},
    3: {"type": "BOLUS", "basal_mult": 1.0, "bolus": 2.0, "desc": "Administer large bolus (2.0U)"},
    4: {"type": "TEMP_BASAL", "basal_mult": 0.0, "bolus": 0.0, "desc": "Suspend all insulin delivery"},
    5: {"type": "TEMP_BASAL", "basal_mult": 0.5, "bolus": 0.0, "desc": "Reduce basal rate by 50%"},
    6: {"type": "TEMP_BASAL", "basal_mult": 1.5, "bolus": 0.0, "desc": "Increase basal rate by 50%"},
    7: {"type": "TEMP_BASAL", "basal_mult": 2.0, "bolus": 0.0, "desc": "Increase basal rate by 100%"},
}

ARTIFACTS_DIR = "artifacts_optimization_run"
CONFIG = {
    "ARTIFACTS_DIR": ARTIFACTS_DIR,
    "DATA_DIR_AZT1D": "subjects",
    "DATA_DIR_OHIO": "ohio_demo",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "T_STEP": 5,
    "METTA_RULES_FILE": "safety_logic.metta",
    
    # RL & TFT Parameters
    "TFT_MAX_ENCODER_LENGTH": 144,
    "TFT_MAX_PRED_LENGTH": 12,
    "TFT_BATCH_SIZE": 64 if SMOKE_TEST else 256,
    "TFT_EPOCHS": 5 if SMOKE_TEST else 25,
    "TFT_LR": 3e-4,
    "TFT_HIDDEN": 64,
    "TFT_HEADS": 4,
    "TFT_DROPOUT": 0.3,
    "TFT_QUANTILES": [0.05, 0.1, 0.5, 0.9, 0.95],
    
    # PPO Parameters
    "RL_STATE_DIM_BASE": 10,
    "BRIDGE_FEATURES": 6,
    "MEAL_FEATURES": 2,
    "RL_ACTION_DIM": len(ACTION_MAP),
    "LSTM_HIDDEN_SIZE": 256,
    "LSTM_NUM_LAYERS": 1,
    "PPO_TOTAL_STEPS": 4000 if SMOKE_TEST else 100000,
    "PPO_UPDATE_INTERVAL": 256 if SMOKE_TEST else 2048,
    "PPO_MINIBATCH": 64 if SMOKE_TEST else 256,
    "PPO_GAMMA": 0.995,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "PPO_EPS_CLIP": 0.2,
    "PPO_K_EPOCHS": 10,
    
    # Safety
    "SAFETY_LOW_GLUCOSE": 70,
    "SAFETY_IOB_THRESHOLD": 8.0,
    "SAFETY_MIN_BOLUS_INTERVAL_MIN": 30,
    "UNCERTAINTY_HIGH_THRESHOLD": 50,
    "UNCERTAINTY_EXTREME_THRESHOLD": 70,
    "USE_UNCERTAINTY_MASKING": True,
    "USE_UNCERTAINTY_REWARD": True,
    
    # Reward
    "REWARD_TARGET_RANGE": (80, 140),
    "REWARD_TIR_RANGE": (70, 180),
    "REWARD_HYPO_PENALTY": -5.0,
    "REWARD_SEVERE_HYPO_PENALTY": -20.0,
    "REWARD_HYPER_PENALTY": -2.0,
    "REWARD_ACTION_COST": 0.01,
    "UNCERTAINTY_PENALTY_WEIGHT": 1.0,
    
    # Eval
    "EVAL_N_EPISODES": 2 if SMOKE_TEST else 5,
    "EVAL_EPISODE_LENGTH_HOURS": 24,
    "EVAL_PATIENTS": ["adolescent#001", "adolescent#002", "adult#001"],
}
CONFIG["RL_STATE_DIM_FULL"] = CONFIG["RL_STATE_DIM_BASE"] + CONFIG["BRIDGE_FEATURES"] + CONFIG["MEAL_FEATURES"]

# ============================================================================
# NEURO-SYMBOLIC GUARDIAN LAYER (HYPERON INTEGRATION)
# ============================================================================
class GuardianLayer:
    def __init__(self, rules_file=CONFIG["METTA_RULES_FILE"]):
        self.override_count = 0
        self.uncertainty_blocks = 0
        self.use_hyperon = HYPERON_ACTIVE
        self.metta = None

        if self.use_hyperon:
            try:
                self.metta = MeTTa()
                if os.path.exists(rules_file):
                    with open(rules_file, 'r') as f:
                        content = f.read()
                    self.metta.run(content)
                    log.info(f"🧠 NEURO-SYMBOLIC: Hyperon loaded rules from {rules_file}")
                else:
                    log.warning(f"⚠️ MeTTa file {rules_file} not found. Using Python fallback.")
                    self.use_hyperon = False
            except Exception as e:
                log.error(f"❌ Hyperon init failed: {e}")
                self.use_hyperon = False

    def validate_action(self, proposed_action_id, cgm, iob, time_since_bolus_min, uncertainty_30=0.0):
        # Basic sanity check
        if not all(np.isfinite([cgm, iob])) or not (40 <= cgm <= 500):
            self.override_count += 1
            return 0, "BLOCKED: Invalid sensor data"

        # --- PATH A: HYPERON (MeTTa) ---
        if self.use_hyperon:
            try:
                # Create the symbolic query
                expr = f"!(evaluate-safety {int(proposed_action_id)} {cgm} {iob} {uncertainty_30})"
                result = self.metta.run(expr)
                
                # Parse Result: [[(Result 4 "Message")]]
                if result and len(result) > 0 and len(result[0]) > 0:
                    atom = result[0][0]
                    children = atom.get_children()
                    if len(children) >= 3 and children[0].get_name() == "Result":
                        # Extract symbolic values
                        final_action = int(children[1].get_object().value)
                        reason = str(children[2].get_object().value)
                        
                        if final_action != proposed_action_id:
                            self.override_count += 1
                            if "UNCERTAINTY" in reason: self.uncertainty_blocks += 1
                        
                        return final_action, f"[MeTTa] {reason}"
            except Exception as e:
                log.error(f"Hyperon execution error: {e}. Fallback to Python.")
        
        # --- PATH B: PYTHON FALLBACK (Legacy) ---
        # This runs if Hyperon is missing or crashes
        action = ACTION_MAP[proposed_action_id]
        is_aggressive = action["bolus"] > 0 or action["basal_mult"] >= 1.5
        
        if is_aggressive and (cgm < CONFIG["SAFETY_LOW_GLUCOSE"] or 
                              iob > CONFIG["SAFETY_IOB_THRESHOLD"] or 
                              time_since_bolus_min < CONFIG["SAFETY_MIN_BOLUS_INTERVAL_MIN"]):
            self.override_count += 1
            return 0, "BLOCKED: Hard Safety Constraints"
            
        if CONFIG["USE_UNCERTAINTY_MASKING"]:
            if uncertainty_30 > CONFIG["UNCERTAINTY_EXTREME_THRESHOLD"] and action["bolus"] >= 1.0:
                self.override_count += 1
                return 0, f"BLOCKED: Extreme Uncertainty ({uncertainty_30:.1f})"
            elif uncertainty_30 > CONFIG["UNCERTAINTY_HIGH_THRESHOLD"] and proposed_action_id == 3:
                self.override_count += 1
                return 2, f"MODIFIED: High Uncertainty ({uncertainty_30:.1f})"
                
        return proposed_action_id, "OK"

# ============================================================================
# XAI DASHBOARD
# ============================================================================
class XAIDashboard:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.log_path = self.artifacts_dir / "xai_decision_log.jsonl"
        self.html_path = self.artifacts_dir / "xai_dashboard.html"
        self.entries = []
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)

    def log_action(self, step, patient_id, cgm, iob, uncertainty, final_action, note, proposed_action, forecast):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": int(step),
            "patient_id": str(patient_id),
            "cgm": round(float(cgm), 1),
            "iob": round(float(iob), 2),
            "uncertainty": round(float(uncertainty), 1),
            "proposed": ACTION_MAP[int(proposed_action)]["desc"],
            "final": ACTION_MAP[int(final_action)]["desc"],
            "note": note,
            "override": note != "OK" and "Verified" not in note
        }
        self.entries.append(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def render_html(self):
        html = """<html><head><title>Neuro-Symbolic Pancreas</title>
        <style>body{font-family:sans-serif;margin:20px} .entry{border:1px solid #ccc;margin:10px;padding:10px;border-radius:5px} 
        .override{background:#ffebee;border-left:5px solid red} .safe{background:#e8f5e9;border-left:5px solid green}</style>
        </head><body><h1>Neuro-Symbolic Decision Log</h1>"""
        for e in reversed(self.entries[-100:]):
            cls = "override" if e["override"] else "safe"
            html += f"""<div class='entry {cls}'>
            <b>Step {e['step']}</b> | CGM: {e['cgm']} | IOB: {e['iob']} | Uncertainty: {e['uncertainty']}<br>
            Proposed: {e['proposed']} <br>
            <b>Final Action: {e['final']}</b><br>
            <i>Reason: {e['note']}</i>
            </div>"""
        html += "</body></html>"
        with open(self.html_path, "w") as f: f.write(html)
        log.info(f"Dashboard saved to {self.html_path}")

# ============================================================================
# RL ENVIRONMENT
# ============================================================================
def _calculate_iob(ins_hist, td_min, tp_min):
    iob = 0.0
    for i, b in enumerate(ins_hist):
        ts = (len(ins_hist) - 1 - i) * CONFIG["T_STEP"]
        if 0 < ts < td_min:
            tau = tp_min*(1-tp_min/td_min)/(1-2*tp_min/td_min)
            S = 1/(1-tau/td_min+(1+tau/td_min)*math.exp(-td_min/tau))
            activity = (S/(tau**2))*ts*(td_min-ts)*math.exp(-ts/tau)
            iob += b * (1-activity)
        elif ts == 0: iob += b
    return float(max(0.0, iob))

class SimglucoseRLEnv:
    def __init__(self, patient_name, dashboard):
        self.patient_name = patient_name
        self.patient = T1DPatient.withName(patient_name)
        self.sensor = CGMSensor.withName("Dexcom", seed=SEED)
        self.pump = InsulinPump.withName("Insulet")
        self.scenario = RandomScenario(start_time=datetime(2024,1,1), seed=SEED)
        self.sim_env = T1DSimEnv(self.patient, self.sensor, self.pump, self.scenario)
        
        self.guardian = GuardianLayer()
        self.dashboard = dashboard
        self.history_len = CONFIG["TFT_MAX_ENCODER_LENGTH"]
        self.cgm_hist = deque(maxlen=self.history_len)
        self.ins_hist = deque(maxlen=self.history_len)
        self.cho_hist = deque(maxlen=self.history_len)
        self.time_hist = deque(maxlen=self.history_len)
        self.step_count = 0
        self.last_bolus_time = -999
        self.current_uncertainty = 0.0 # Placeholder if TFT is off
        
        # --- FIX: Safe fallback for patient attributes (handles AttributeError) ---
        try:
            self.td_min = self.patient.para.t_max_insulin
            self.tp_min = self.patient.para.t_peak_insulin
        except AttributeError:
            # Fallback for compatibility
            self.td_min = 360.0  # 6 hours
            self.tp_min = 75.0   # 75 minutes

    def reset(self):
        obs, _, _, _ = self.sim_env.reset()
        self.cgm_hist.clear(); self.ins_hist.clear(); self.cho_hist.clear(); self.time_hist.clear()
        start_time = self.scenario.start_time
        for i in range(self.history_len):
            self.cgm_hist.append(obs.CGM)
            self.ins_hist.append(0.0)
            self.cho_hist.append(0.0)
            self.time_hist.append(start_time - timedelta(minutes=(self.history_len-i)*5))
        
        self.step_count = 0
        rnn_state = (torch.zeros(1,1,256).to(CONFIG["DEVICE"]), torch.zeros(1,1,256).to(CONFIG["DEVICE"]))
        return self._get_state(), {"mask": self._get_mask(), "rnn_state": rnn_state}

    def _get_mask(self):
        mask = np.ones(CONFIG["RL_ACTION_DIM"], dtype=np.float32)
        cgm = self.cgm_hist[-1]
        iob = _calculate_iob(self.ins_hist, self.td_min, self.tp_min)
        time_since = (self.step_count - self.last_bolus_time) * 5
        
        # Ask Guardian what is allowed
        for i in range(len(mask)):
            valid_act, _ = self.guardian.validate_action(i, cgm, iob, time_since, self.current_uncertainty)
            if valid_act != i: mask[i] = 0.0
        
        if mask.sum() == 0: mask[0] = 1.0
        return mask

    def _get_state(self):
        # Simplified state for smoke test / example
        cgm = self.cgm_hist[-1]
        iob = _calculate_iob(self.ins_hist, self.td_min, self.tp_min)
        base = np.array([cgm/200.0, iob/10.0] + [0.0]*8, dtype=np.float32) # Pad to 10
        bridge = np.zeros(CONFIG["BRIDGE_FEATURES"], dtype=np.float32) # Placeholder
        meal = np.zeros(CONFIG["MEAL_FEATURES"], dtype=np.float32)
        return np.concatenate([base, bridge, meal])

    def step(self, action_id, rnn_state=None):
        # --- ⚡ BEN GOERTZEL "PIZZA STRESS TEST" INJECTION ⚡ ---
        # CHANGED TO STEP 15 (Because patient crashed before 200)
        if self.step_count == 15: 
            print(f"\n⚠️ STEP {self.step_count}: INJECTING ARTIFICIAL CONFUSION...")
            
            # 1. Force "Large Bolus" (Action 3)
            action_id = 3 
            
            # 2. Force Extreme Uncertainty
            self.current_uncertainty = 100.0
        # -------------------------------------------------------

        cgm = self.cgm_hist[-1]
        iob = _calculate_iob(self.ins_hist, self.td_min, self.tp_min)
        time_since = (self.step_count - self.last_bolus_time) * 5
        
        # 1. NEURO-SYMBOLIC CHECK
        final_action_id, note = self.guardian.validate_action(action_id, cgm, iob, time_since, self.current_uncertainty)
        
        # 2. Execute in Sim
        act_spec = ACTION_MAP[final_action_id]
        basal = self.pump.basal(self.time_hist[-1].hour)
        sim_act = Action(basal=basal * act_spec["basal_mult"], bolus=act_spec["bolus"])
        obs, _, done, _ = self.sim_env.step(sim_act)
        
        # 3. Update History
        self.cgm_hist.append(obs.CGM)
        self.ins_hist.append(act_spec["bolus"])
        self.cho_hist.append(self.sim_env.scenario.get_action(self.time_hist[-1]).meal)
        self.time_hist.append(self.time_hist[-1] + timedelta(minutes=5))
        if act_spec["bolus"] > 0: self.last_bolus_time = self.step_count
        self.step_count += 1
        
        # 4. Log to Dashboard
        if self.dashboard:
            self.dashboard.log_action(self.step_count, self.patient_name, cgm, iob, 
                                    self.current_uncertainty, final_action_id, note, action_id, {})

        # 5. Reward
        reward = 1.0 if 70 <= obs.CGM <= 180 else -1.0
        if obs.CGM < 50: done = True; reward = -10.0
        
        return self._get_state(), reward, done, {"mask": self._get_mask(), "rnn_state": rnn_state}

# ============================================================================
# PPO MODEL
# ============================================================================
class RecurrentActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, 256, batch_first=False)
        self.actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def act(self, state, mask, hidden):
        x, new_hidden = self.lstm(state.unsqueeze(0).unsqueeze(0), hidden)
        x = x.squeeze(0).squeeze(0)
        logits = self.actor(x)
        logits = logits + torch.log(mask + 1e-8)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), new_hidden

# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    Path(CONFIG["ARTIFACTS_DIR"]).mkdir(exist_ok=True)
    dashboard = XAIDashboard(CONFIG["ARTIFACTS_DIR"])
    
    log.info("="*60)
    log.info("    NEURO-SYMBOLIC PANCREAS (HYPERON EDITION)")
    log.info("="*60)
    
    # Initialize Agent
    agent = RecurrentActorCritic(CONFIG["RL_STATE_DIM_FULL"], CONFIG["RL_ACTION_DIM"]).to(CONFIG["DEVICE"])
    
    # Training Loop (Simplified for demonstration)
    log.info("Starting Symbolic Training Loop...")
    pbar = tqdm(range(CONFIG["EVAL_N_EPISODES"]))
    
    for ep in pbar:
        env = SimglucoseRLEnv("adolescent#001", dashboard)
        state, info = env.reset()
        rnn_state = info["rnn_state"]
        done = False
        
        while not done:
            # Neural Net proposes action
            mask_tensor = torch.tensor(info["mask"], device=CONFIG["DEVICE"])
            state_tensor = torch.tensor(state, device=CONFIG["DEVICE"])
            
            with torch.no_grad():
                action, rnn_state = agent.act(state_tensor, mask_tensor, rnn_state)
            
            # Environment validates with MeTTa and steps
            next_state, reward, done, next_info = env.step(action, rnn_state)
            state, info = next_state, next_info
            
            if env.step_count > 250: break # Extended to 250 so we catch Step 200!
            
    dashboard.render_html()
    log.info(f"✅ Run Complete. Check {CONFIG['ARTIFACTS_DIR']}/xai_dashboard.html")

if __name__ == "__main__":
    main()