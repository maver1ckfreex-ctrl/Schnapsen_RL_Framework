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
import torch.optim as optim
import torch.multiprocessing as mp
import pathlib
import random
import time
import os
import sys
import csv
from collections import deque
import queue

# --- PATH SETUP ---
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent)) 
from schnapsen.game import SchnapsenGamePlayEngine, Bot
from schnapsen.bots import RdeepBot
from schnapsen.bots.ml_bot import get_state_feature_vector, get_move_feature_vector

# --- CONFIGURATION---
# 1.--- CONFIGURATION (depth)---
# Renamed for Control Experiment
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_dp2.pth"
MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_dp4.pth"  #default depth
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_dp6.pth"
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_dp8.pth"
# Training log
#CSV_LOG_PATH = pathlib.Path("RL_dp2.csv")
CSV_LOG_PATH = pathlib.Path("RL_dp4.csv")                       #default depth  
#CSV_LOG_PATH = pathlib.Path("RL_dp6.csv")
#CSV_LOG_PATH = pathlib.Path("RL_dp8.csv")   
  
# 2.--- CONFIGURATION (samples)---
# Renamed for Control Experiment(samples)
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_s4.pth"   #default samples
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_s20.pth"  
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_s40.pth"
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_s60.pth"
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "RL_s80.pth"
# Training log
#CSV_LOG_PATH = pathlib.Path("RL_s4.csv")                       #default samples
#CSV_LOG_PATH = pathlib.Path("RL_s20.csv")                         
#CSV_LOG_PATH = pathlib.Path("RL_s40.csv")
#CSV_LOG_PATH = pathlib.Path("RL_s60.csv")   
#CSV_LOG_PATH = pathlib.Path("RL_s80.csv")  
              
"""!!!REMEMBER ADJUST THE DEPTH OR SAMPLES OF RDEEP BOT WHEN YOU WANT TRAINING DIFFERENT BOT(LINE 117) !!!"""
INPUT_DIM = 173
HIDDEN_DIM = 512         

# Settings (Same as treatment group)
TOTAL_GAMES = 1200000         
BATCH_SIZE = 1024             
TARGET_UPDATE_FREQ = 2000     
MLP_LR = 0.0003               

# --- DECAY SETTINGS ---
EPSILON_START = 0.23
EPSILON_END = 0.02

# --- 1. ARCHITECTURE: Standard MLP (Control) ---
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),     #Hidden Layer  (Linear)
            nn.ReLU(),                            #Hidden Layer  (Activation)
            nn.Linear(HIDDEN_DIM, 1),             #Output Layer  (Linear)
            nn.Sigmoid()                          #Output Layer  (Activation)
        )

    def forward(self, x):
        # Direct mapping: State -> Probability
        return self.net(x)

# --- 2. WORKER PROCESS ---
def worker_loop(worker_id, state_dict_queue, data_queue):
    local_model = StandardMLP().to("cpu")
    local_model.eval()
    engine = SchnapsenGamePlayEngine()
    
    current_epsilon = EPSILON_START

    try:
        msg = state_dict_queue.get(timeout=10)
        if isinstance(msg, tuple):
            local_model.load_state_dict(msg[0])
            current_epsilon = msg[1]
        else:
            local_model.load_state_dict(msg)
    except:
        pass

    class AsyncBot(Bot):
        def __init__(self):
            super().__init__("Hero")
            self.game_history = []
        
        def get_move(self, perspective, leader_move):
            valid_moves = perspective.valid_moves()
            state_vec = get_state_feature_vector(perspective)
            
            if random.random() < current_epsilon: 
                 chosen_move = random.choice(valid_moves)
            else:
                candidates = []
                for move in valid_moves:
                    move_vec = get_move_feature_vector(move)
                    if perspective.am_i_leader():
                        full_input = state_vec + move_vec + get_move_feature_vector(None)
                    else:
                        full_input = state_vec + get_move_feature_vector(leader_move) + move_vec
                    candidates.append(full_input)
                
                with torch.no_grad():
                    inp_tensor = torch.tensor(candidates, dtype=torch.float32)
                    scores = local_model(inp_tensor)
                
                best_idx = torch.argmax(scores).item()
                chosen_move = valid_moves[best_idx]
            
            if perspective.am_i_leader():
                final_input = state_vec + get_move_feature_vector(chosen_move) + get_move_feature_vector(None)
            else:
                final_input = state_vec + get_move_feature_vector(leader_move) + get_move_feature_vector(chosen_move)
            
            self.game_history.append(final_input)
            return chosen_move

    games_played = 0
    
    while True:
        if not state_dict_queue.empty():
            try:
                while not state_dict_queue.empty():
                    msg = state_dict_queue.get_nowait()
                    if isinstance(msg, tuple):
                        local_model.load_state_dict(msg[0])
                        current_epsilon = msg[1]
                    else:
                        local_model.load_state_dict(msg)
            except queue.Empty: pass

        hero = AsyncBot()
        villain = RdeepBot(num_samples=4, depth=4, rand=random.Random(time.time() + worker_id)) #default depth 4
        rng = random.Random()
        
        if games_played % 2 == 0:
            winner, _, _ = engine.play_game(hero, villain, rng)
        else:
            winner, _, _ = engine.play_game(villain, hero, rng)
            
        did_win = (str(winner).strip() == "Hero")
        reward = 1.0 if did_win else 0.0
        
        try:
            data_queue.put((hero.game_history, reward, 1 if did_win else 0))
        except: pass
            
        games_played += 1

# --- 3. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Turbo Async Training on {device}")
    
    policy_net = StandardMLP().to(device)
    
    if MODEL_SAVE_PATH.exists():
        print("Resuming...")
        policy_net.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    optimizer = optim.Adam(policy_net.parameters(), lr=MLP_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    cpu_count = os.cpu_count() or 1
    num_workers = max(1, cpu_count - 2)
    
    worker_queues = [mp.Queue() for _ in range(num_workers)]
    data_queue = mp.Queue(maxsize=10000)
    
    print(f"--- Launching {num_workers} Async Workers ---")
    
    workers = []
    initial_weights = {k: v.cpu() for k, v in policy_net.state_dict().items()}
    initial_msg = (initial_weights, EPSILON_START)
    
    for i in range(num_workers):
        worker_queues[i].put(initial_msg)
        p = mp.Process(target=worker_loop, args=(i, worker_queues[i], data_queue))
        p.start()
        workers.append(p)
        
    replay_buffer = []
    MAX_BUFFER = 100000
    total_games = 0
    total_wins = 0
    win_history = deque(maxlen=200)
    
    steps_since_target_update = 0
    last_update_time = time.time()
    
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(["Games", "WinRate_200", "Total_WR", "Loss", "Epsilon", "Time"])
    
    try:
        while total_games < TOTAL_GAMES:
            games_fetched = 0
            while not data_queue.empty() and games_fetched < 50:
                try:
                    history, reward, is_win = data_queue.get_nowait()
                    for state in history:
                        replay_buffer.append((state, reward))
                    total_games += 1
                    total_wins += is_win
                    win_history.append(is_win)
                    games_fetched += 1
                except queue.Empty: break
            
            if len(replay_buffer) > MAX_BUFFER:
                replay_buffer = replay_buffer[-MAX_BUFFER:]
            
            current_loss = 0.0
            if len(replay_buffer) > BATCH_SIZE * 2:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states, rewards = zip(*batch)
                """State tensor is a tensor matrix R^1024(batch size)x173(feature vector dimension)"""
                s_tensor = torch.tensor(states, dtype=torch.float32).to(device)                   #State Tensor
                """Reward tensor is a tensor matrix R^1024(batch size)x1(result 1.0 win or 0.0 lose ), which records the actually result"""
                r_tensor = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)     #reward tensor
                
                optimizer.zero_grad()
                preds = policy_net(s_tensor)
                
                loss = nn.MSELoss()(preds, r_tensor)
                loss.backward()
                optimizer.step()
                current_loss = loss.item()
                
                steps_since_target_update += 1
                
                if steps_since_target_update >= TARGET_UPDATE_FREQ:
                    steps_since_target_update = 0
                    
                    scheduler.step(current_loss)

                    progress = min(1.0, total_games / TOTAL_GAMES)
                    curr_eps = EPSILON_START - (progress * (EPSILON_START - EPSILON_END))
                    
                    cpu_weights = {k: v.cpu() for k, v in policy_net.state_dict().items()}
                    msg = (cpu_weights, curr_eps)
                    
                    for q in worker_queues:
                        while not q.empty():
                            try: q.get_nowait()
                            except: pass
                        q.put(msg)
                        
                    print(f"  [Sync] Updated Workers | Eps: {curr_eps:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if total_games % 100 == 0 and games_fetched > 0:
                elapsed = time.time() - last_update_time
                if elapsed > 1.0:
                    wr_200 = (sum(win_history) / len(win_history)) * 100 if win_history else 0
                    total_wr = (total_wins / total_games) * 100
                    
                    progress = min(1.0, total_games / TOTAL_GAMES)
                    disp_eps = EPSILON_START - (progress * (EPSILON_START - EPSILON_END))
                    
                    print(f"Games: {total_games} | WR_200: {wr_200:.1f}% | Total: {total_wr:.1f}% | Eps: {disp_eps:.3f}", end='\r')
                    
                    with open(CSV_LOG_PATH, 'a', newline='') as f:
                        csv.writer(f).writerow([total_games, f"{wr_200:.2f}", f"{total_wr:.2f}", f"{current_loss:.4f}", f"{disp_eps:.4f}", f"{time.time()}"])
                    
                    last_update_time = time.time()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved to {MODEL_SAVE_PATH}")
