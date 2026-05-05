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
from torch.utils.data import TensorDataset, DataLoader
import pathlib
import time
import os
import math
import concurrent.futures

# --- CONFIGURATION ---(default depth=4)
DATA_PATH = "ML_data_hub"
#DATA_FILE = "rdeep_rdeep_20k_games_dp2.txt"
DATA_FILE = "rdeep_rdeep_20k_games_dp4.txt" 
#DATA_FILE = "rdeep_rdeep_20k_games_dp6.txt"
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "MLP_dp2.pth"
MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "MLP_dp4.pth" 
#MODEL_SAVE_PATH = pathlib.Path("ML_model_hub") / "MLP_dp6.pth"

INPUT_DIM = 173
BATCH_SIZE = 64
LR = 0.0005
EPOCHS = 100

# --- DEVICE ---
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- THE MODEL (Shallow MLP) ---
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Classifier: INPUT_DIM -> 512 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),      #Hidden Layer  (Linear)
            nn.ReLU(),                      #Hidden Layer  (Activation)                   
            nn.Linear(512, 1)               #Output Layer  (Linear)
        )
        self.sigmoid = nn.Sigmoid()         #Output Layer  (Activation)

    def forward(self, x):
        features = self.classifier(x)
        prediction = self.sigmoid(features)
        return prediction 

# --- PARALLEL DATA PARSING WORKER ---
def parse_chunk(lines):
    """
    Parses text lines. 
    Format: "FeatureVector(X) || Label"(Y) (2 parts)
    """
    chunk_X, chunk_Y = [], [] 
    for line in lines:
        try:
            parts = line.strip().split("||")
            
            if len(parts) != 2: continue 
            
            feat_str, label_str = parts 
            
            chunk_X.append([float(x) for x in feat_str.split(',')])
            chunk_Y.append([float(label_str)])
        except ValueError:
            continue
    return chunk_X, chunk_Y 

def load_data_parallel():
    file_path = pathlib.Path(DATA_PATH) / DATA_FILE
    if not file_path.exists():
        print("Data file not found. Run cli_c.py first.")
        return None

    print(f"Reading file from disk...")
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
        
    total_lines = len(all_lines)
    if total_lines == 0: return None
    
    cpu_count = os.cpu_count() or 1
    print(f"Parsing {total_lines} lines using {cpu_count} CPU cores...")
    
    chunk_size = math.ceil(total_lines / cpu_count)
    chunks = [all_lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
    
    X_final, Y_final = [], [] 
    
    start_parse = time.time()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(parse_chunk, chunks)
        
        for res_X, res_Y in results: 
            X_final.extend(res_X)
            Y_final.extend(res_Y)
            
    print(f"Data parsing finished in {time.time() - start_parse:.2f}s")
    
    if len(X_final) == 0:
        print("❌ Error: No data parsed! Check if your delimiter is exactly '||' and lines have 2 parts.")
        return None

    print("Moving data to GPU (Tensor conversion)...")
    
    X = torch.tensor(X_final, dtype=torch.float32).to(device)
    Y = torch.tensor(Y_final, dtype=torch.float32).to(device)
    
    return TensorDataset(X, Y)

def train():
    dataset = load_data_parallel()
    if dataset is None: return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = StandardMLP().to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    criterion_pred = nn.BCELoss()

    print(f"--- STARTING TRAINING (Standard MLP Control) ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_pred_loss = 0
        
        for batch_X, batch_Y in dataloader: 
            optimizer.zero_grad()
            
            # Forward Pass
            pred = model(batch_X) 
            
            # --- CALCULATE LOSSES ---
            loss = criterion_pred(pred, batch_Y)
            
            loss.backward()
            optimizer.step()
            
            total_pred_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            avg_pred = total_pred_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{EPOCHS}: Pred_BCE={avg_pred:.4f}")

    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f}s")
    
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
