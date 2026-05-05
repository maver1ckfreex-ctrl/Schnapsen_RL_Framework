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

import random
import pathlib

from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import shutil
import multiprocessing

import click

from schnapsen.bots import MLDataBot


from schnapsen.game import (Bot, GamePlayEngine, Move, PlayerPerspective,
                            SchnapsenGamePlayEngine, TrumpExchange)
from schnapsen.alternative_engines.twenty_four_card_schnapsen import TwentyFourSchnapsenGamePlayEngine

from schnapsen.bots.rdeep import RdeepBot


@click.group()
def main() -> None:
    """Various Schnapsen Game Examples"""


@main.group()
def ml() -> None:
    """Commands for the ML bot"""

    
    
@ml.command()
def create_replay_dataset_parallel() ->None:
    # define replay memory database creation parameters
    num_of_games: int = 20000
    replay_memory_dir: str = 'ML_data_hub'
    #replay_memory_filename: str = 'rdeep_rdeep_20k_games_dp2.txt'
    replay_memory_filename: str = 'rdeep_rdeep_20k_games_dp4.txt'       #default depth
    #replay_memory_filename: str = 'rdeep_rdeep_20k_games_dp6.txt'
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename
    
    # Define temp directory for parallel workers
    temp_dir = pathlib.Path(replay_memory_dir) / "temp_parallel_data"

    delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
    if delete_existing_older_dataset and replay_memory_location.exists():
        print(f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected.")
        replay_memory_location.unlink()

    # in any case make sure the directory exists
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --- Parallel Execution Setup ---
    # Determine number of CPUs to use (leave one free for system stability if desired, or use all)
    num_workers = multiprocessing.cpu_count()
    print(f"Starting parallel processing on {num_workers} cores...")

    # Calculate chunks (split 20,000 games among workers)
    # e.g., if 4 cores, we get ranges: [1-5000], [5001-10000], etc.
    games_per_worker = num_of_games // num_workers
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            # Define the start and end game IDs for this worker
            start_game = i * games_per_worker + 1
            # If it's the last worker, ensure it grabs any remainder games
            end_game = (i + 1) * games_per_worker + 1 if i < num_workers - 1 else num_of_games + 1
            game_indices = range(start_game, end_game)
            
            # Create a unique temp filename for this worker (e.g., part_0.txt)
            temp_file = temp_dir / f"part_{i}.txt"
            
            # Pass original seeds to bots. 
            # Note: Each worker gets a fresh bot with these seeds. 
            # Diversity is maintained because 'engine.play_game' uses 'random.Random(i)' for the deal.
            futures.append(
                executor.submit(
                    _worker_simulation, 
                    game_indices, 
                    temp_file, 
                    4564654644, # Bot 1 seed
                    68438       # Bot 2 seed
                )
            )

    # Wait for all workers to finish
    for future in futures:
        future.result() # This will raise exceptions if any worker failed

    # --- Merge Results ---
    print("Parallel generation complete. Merging files...")
    with open(replay_memory_location, 'w') as outfile:
        for i in range(num_workers):
            temp_file = temp_dir / f"part_{i}.txt"
            if temp_file.exists():
                with open(temp_file, 'r') as infile:
                    shutil.copyfileobj(infile, outfile)
                temp_file.unlink() # Delete temp file after merging

    # Cleanup temp dir
    if temp_dir.exists():
        temp_dir.rmdir()

    print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}")



# --- Worker Function (Must be defined at module level) ---
def _worker_simulation(game_indices, temp_file_path, bot1_seed, bot2_seed):
    """
    Runs a batch of games on a separate CPU core and writes to a temp file.
    """
    # We re-instantiate the bots here so each core has its own independent bot objects.
    # We use the seeds provided to ensure the logic remains consistent with your original config.
    bot_1_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(bot1_seed)) # default depth=4 you can set 2 or 6
    bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(bot2_seed)) # default depth=4 you can set 2 or 6

    # Ensure the directory for the temp file exists
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Engine and Bots pointing to the specific TEMP file for this core
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_location=temp_file_path)
    replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_location=temp_file_path)

    # Run the batch of games
    for i in game_indices:
        engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(i))
        
    return f"Finished batch of {len(game_indices)} games."
    

if __name__ == "__main__":
    main()
