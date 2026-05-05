"""
--- IMPORTANT SERVER CONFIGURATION NOTE ---
1.If you are running this on a High-Performance Server (HPC) or a machine with many cores 
(e.g., 64+), enabling the lines below is CRITICAL to prevent crashes.

The Issue: 
Standard libraries like NumPy/PyTorch try to use all available CPU cores for matrix math. 
When combined with Python's multiprocessing (launching 60+ workers), this creates an 
explosion of threads (e.g., 60 workers * 64 threads = ~3800 threads), which hits the 
OS 'ulimit' and causes "pthread_create failed" errors.

The Fix: 
We force the underlying math libraries to use only 1 thread per worker. This allows 
us to maximize the number of *worker processes* (agents playing games) without 
overloading the system scheduler.
2. MAKE SURE YOU CREATE THE ML_data_hub folder AT Schnapsen folder TO SAVE THE MODEL
"""

"""  # <--- DELETE THIS LINE (triple quotes) if you are running on a SERVER
import os

# --- CRITICAL FIX FOR SERVERS ---
# Force libraries to use only 1 thread per worker process to prevent thread explosion.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -------------------------------
"""  # <--- DELETE THIS LINE (triple quotes) if you are running on a SERVER

import torch
import torch.nn as nn
import pathlib
import random
import sys
import concurrent.futures
import os
import time
import csv
from collections import defaultdict

# --- PATH SETUP ---
root_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from schnapsen.game import SchnapsenGamePlayEngine, Bot, PlayerPerspective, Move
from schnapsen.bots import RdeepBot
from schnapsen.bots.ml_bot import get_state_feature_vector, get_move_feature_vector

# --- CONFIGURATION ---
GAMES_PER_MATCHUP = 10000 
CHUNK_SIZE = 500  # Split tasks for better CPU utilization
MODEL_DIR = pathlib.Path("ML_model_hub")
MATRIX_FILE = "winning_rate_matrix_10k_RL_samples.csv"

INPUT_DIM = 173
HIDDEN_DIM=512
CLASSIFIER_HIDDEN = 512

# --- 1. MODEL ARCHITECTURES ---

class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # REMOVED: Expansion / Intelligence Vector logic
        # REPLACED: Simple 2-layer Perceptron
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(), 
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Direct mapping: State -> Probability
        return self.net(x)




# --- 2. UNIVERSAL BOT LOADER ---
class PyTorchBot(Bot):
    def __init__(self, name, model_class, model_path):
        super().__init__(name)
        self.device = torch.device("cpu") # CPU is safer for multiprocessing
        self.model = model_class().to(self.device)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # weights_only=True prevents security warnings
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def get_move(self, perspective, leader_move):
        valid_moves = perspective.valid_moves()
        state_vec = get_state_feature_vector(perspective)
        
        candidates = []
        for move in valid_moves:
            move_vec = get_move_feature_vector(move)
            if perspective.am_i_leader():
                full = state_vec + move_vec + get_move_feature_vector(None)
            else:
                full = state_vec + get_move_feature_vector(leader_move) + move_vec
            candidates.append(full)
            
        with torch.no_grad():
            input_tensor = torch.tensor(candidates, dtype=torch.float32).to(self.device)
            out = self.model(input_tensor)
            
            # IntelligenceNet returns (vector, prob), StandardMLP returns (prob)
            # We handle both cases here:
            if isinstance(out, tuple):
                probs = out[1] # Take prediction
            else:
                probs = out
            
        best_idx = torch.argmax(probs).item()
        return valid_moves[best_idx]

# --- 3. WORKER FUNCTION ---
def play_matchup_chunk(args):
    """
    Plays a small chunk of games between two specific bots.
    """
    p1_config, p2_config, seed_start, n_games = args
    
    def load_bot_from_config(cfg, seed_offset):
        rng = random.Random(seed_offset)
        if cfg['type'] == 'rdeep':
            return RdeepBot(depth=4, num_samples=cfg['samples'], rand=rng, name=cfg['name'])
        elif cfg['type'] == 'RL':
            return PyTorchBot(cfg['name'], StandardMLP, cfg['path'])
 
        raise ValueError(f"Unknown bot type: {cfg['type']}")

    # Load Bots with distinct seeds
    try:
        bot1 = load_bot_from_config(p1_config, seed_start)
        bot2 = load_bot_from_config(p2_config, seed_start + 999999)
    except Exception as e:
        return (p1_config['name'], p2_config['name'], 0, 0, f"Load Error: {e}")

    engine = SchnapsenGamePlayEngine()
    p1_wins = 0
    
    for i in range(n_games):
        game_rng = random.Random(seed_start + i)
        
        # Alternating Start
        if i % 2 == 0:
            winner, _, _ = engine.play_game(bot1, bot2, game_rng)
        else:
            winner, _, _ = engine.play_game(bot2, bot1, game_rng)
            
        # FIX: Check if the winner object is literally bot1
        if winner == bot1:
            p1_wins += 1
            
    return (p1_config['name'], p2_config['name'], p1_wins, n_games, None)

# --- 4. MAIN CONTROLLER ---
def generate_winning_matrix_parallel():
# --- A. DEFINE THE ROSTER ---
    bot_registry_1 = [
        # RL
        {'name': 'RL-S4',        'type': 'RL',  'path': MODEL_DIR / 'RL_s4.pth'},
        {'name': 'RL-S20',        'type': 'RL',  'path': MODEL_DIR / 'RL_s20.pth'},
        {'name': 'RL-S40',        'type': 'RL',  'path': MODEL_DIR / 'RL_s40.pth'},
        {'name': 'RL-S60',        'type': 'RL',  'path': MODEL_DIR / 'RL_s60.pth'},
        {'name': 'RL-S80',        'type': 'RL',  'path': MODEL_DIR / 'RL_s80.pth'},
    ]
    bot_registry_2 = [
        #Rdeep
        {'name': 'Rdeep-S2',        'type': 'rdeep', 'samples': 4},
        {'name': 'Rdeep-S20',        'type': 'rdeep', 'samples': 20},
        {'name': 'Rdeep-S40',        'type': 'rdeep', 'samples': 40},
        {'name': 'Rdeep-S60',        'type': 'rdeep', 'samples': 60},
        {'name': 'Rdeep-S80',        'type': 'rdeep', 'samples': 80},
        
        # RL
        {'name': 'RL-S4',        'type': 'RL',  'path': MODEL_DIR / 'RL_s4.pth'},
        {'name': 'RL-S20',        'type': 'RL',  'path': MODEL_DIR / 'RL_s20.pth'},
        {'name': 'RL-S40',        'type': 'RL',  'path': MODEL_DIR / 'RL_s40.pth'},
        {'name': 'RL-S60',        'type': 'RL',  'path': MODEL_DIR / 'RL_s60.pth'},
        {'name': 'RL-S80',        'type': 'RL',  'path': MODEL_DIR / 'RL_s80.pth'},
    ]
    
    player_names = [b['name'] for b in bot_registry_1]
    opponent_names = [b['name'] for b in bot_registry_2]
    
    
    print(f"--- STARTING MATRIX TOURNAMENT (Parallel) ---")
    print(f"Players: {len(player_names)}")
    print(f"Games/Matchup: {GAMES_PER_MATCHUP}")
    print(f"Models Dir: {MODEL_DIR}")
    
    # --- B. PREPARE JOBS ---
    jobs = []
    job_id = 0
    
    for p1_conf in bot_registry_1:
        for p2_conf in bot_registry_2:
            remaining = GAMES_PER_MATCHUP
            seed_cursor = job_id * 100000 
            
            while remaining > 0:
                current_chunk = min(CHUNK_SIZE, remaining)
                jobs.append((p1_conf, p2_conf, seed_cursor, current_chunk))
                remaining -= current_chunk
                seed_cursor += current_chunk
            
            job_id += 1

    cpu_count = os.cpu_count() or 1
    print(f"Generated {len(jobs)} computation tasks. Using {cpu_count} cores.")
    
    # --- C. EXECUTE PARALLEL ---
    results_agg = defaultdict(lambda: defaultdict(int))
    total_games_agg = defaultdict(lambda: defaultdict(int))
    
    start_time = time.time()
    completed_tasks = 0
    errors = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {executor.submit(play_matchup_chunk, job): job for job in jobs}
        
        for future in concurrent.futures.as_completed(futures):
            p1, p2, wins, count, err = future.result()
            
            if err:
                if err not in errors: print(f"\nExample Error: {err}")
                errors.append(err)
            else:
                results_agg[p1][p2] += wins
                total_games_agg[p1][p2] += count
            
            completed_tasks += 1
            if completed_tasks % 50 == 0:
                elapsed = time.time() - start_time
                progress = (completed_tasks / len(jobs)) * 100
                print(f"Progress: {progress:.1f}% | Tasks: {completed_tasks}/{len(jobs)} | Time: {elapsed:.0f}s", end='\r')

    # --- D. WRITE CSV ---
    print(f"\n\nWriting results to {MATRIX_FILE}...")
    
    with open(MATRIX_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Player_vs_Against"] + opponent_names)
        
        for p1 in player_names:
            row = [p1]
            for p2 in opponent_names:
                wins = results_agg[p1][p2]
                total = total_games_agg[p1][p2]
                rate = (wins / total) if total > 0 else 0.0
                row.append(f"{rate:.4f}")
            writer.writerow(row)

    print(f"Done! Matrix saved.")
    if errors:
        print(f"WARNING: {len(errors)} tasks failed. Check model paths.")

if __name__ == "__main__":
    generate_winning_matrix_parallel()
