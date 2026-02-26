"""
Generate embeddings with label mapping for PCG classification
USING THE OFFICIAL TS2Vec REPOSITORY IMPLEMENTATION

Creates:
    embeddings/pcg_embeddings.npy
    embeddings/pcg_labels.npy
    embeddings/embedding_metadata.pkl
"""

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ts2vec import TS2Vec   

# ====================================================
# PATHS
# ====================================================

PROCESSED_DATA = "data/processed/"
EMBEDDINGS_DIR = "embeddings/"
MODELS_DIR = "models/"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ====================================================
# DEVICE SELECTION
# ====================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ====================================================
# 1. LOAD PREPROCESSED DATA
# ====================================================

def load_preprocessed_data():
    print("Loading preprocessed PCG windows...")

    windows = np.load(os.path.join(PROCESSED_DATA, "combined_windows.npy"))
    labels = np.load(os.path.join(PROCESSED_DATA, "combined_labels.npy"))

    file_map = np.load(os.path.join(PROCESSED_DATA, "combined_file_mapping.npy"))
    dir_map = np.load(os.path.join(PROCESSED_DATA, "combined_directory_mapping.npy"))

    print(f"Loaded {len(windows)} windows, each shape: {windows.shape[1]} samples")
    
    # Show label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "Normal", 1: "Abnormal", -1: "Unknown"}
    for val, count in zip(unique, counts):
        name = label_names.get(val, f"Unknown({val})")
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    return windows, labels, file_map, dir_map

# ====================================================
# 2. DATASET WRAPPER
# ====================================================

class PCGDataset(Dataset):
    def __init__(self, windows, labels, file_map, dir_map):
        self.windows = windows
        self.labels = labels
        self.file_map = file_map
        self.dir_map = dir_map

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(-1)      # (2000, 1)
        label = int(self.labels[idx])
        meta = {
            "file": self.file_map[idx],
            "directory": self.dir_map[idx],
            "idx": idx
        }
        return x, label, meta


# ====================================================
# 3. TRAIN OFFICIAL TS2Vec 
# ====================================================

def train_ts2vec(dataset):
    print("\n" + "="*60)
    print("Training Official TS2Vec Model")
    print("="*60)

    # 1. Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Settings
    BATCH_SIZE = 8
    N_EPOCHS = 10
    
    # Calculate total batches per epoch
    n_samples = len(dataset.windows)
    n_batches = int(np.ceil(n_samples / BATCH_SIZE))
    
    # 3. Define Callbacks (Fixed Signatures)
    pbar = None
    current_epoch = 0  # We must track this manually because the library doesn't pass it

    def iter_logger(model, loss):
        """Runs after every batch"""
        nonlocal pbar
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss:.4f}"})

    def epoch_logger(model, loss):
        """Runs after every epoch"""
        nonlocal pbar, current_epoch
        
        # Close the finished epoch's bar
        if pbar is not None:
            pbar.close()
        
        # Print summary
        tqdm.write(f"✅  [Epoch {current_epoch+1}/{N_EPOCHS}] Completed. Avg Loss: {loss:.6f}")
        
        # Increment manual counter
        current_epoch += 1
        
        # Start a new bar for the next epoch (if not finished)
        if current_epoch < N_EPOCHS:
            pbar = tqdm(total=n_batches, desc=f"Epoch {current_epoch+1}/{N_EPOCHS}", unit="batch")

    # 4. Initialize TS2Vec
    if isinstance(DEVICE, torch.device):
        device_str = str(DEVICE).split(":")[0]
    else:
        device_str = str(DEVICE)

    try:
        encoder = TS2Vec(
            input_dims=1,
            output_dims=128,
            hidden_dims=64,
            depth=10,
            device=device_str,
            lr=0.001,
            batch_size=BATCH_SIZE,
            max_train_length=None,
            temporal_unit=0,
            # Link our fixed loggers:
            after_iter_callback=iter_logger,
            after_epoch_callback=epoch_logger
        )
        print(f"✅ TS2Vec initialized (Batch Size: {BATCH_SIZE})")

    except Exception as e:
        print(f"❌ Error initializing TS2Vec: {e}")
        encoder = TS2Vec(input_dims=1, output_dims=128, device=device_str)

    # 5. Prepare Data
    data = dataset.windows.astype(np.float32)
    if data.ndim == 2:
        data = data.reshape(len(data), -1, 1)

    print(f"\nTraining on {len(data)} windows ({n_batches} batches per epoch)")
    
    # Initialize the VERY FIRST progress bar
    pbar = tqdm(total=n_batches, desc=f"Epoch 1/{N_EPOCHS}", unit="batch")

    # 6. Run Training
    encoder.fit(
        train_data=data,
        n_epochs=N_EPOCHS,
        verbose=False
    )
    
    # Ensure the last bar is closed
    if pbar is not None:
        pbar.close()

    return encoder

# ====================================================
# 4. GENERATE GLOBAL EMBEDDINGS (CORRECTED)
# ====================================================
def generate_embeddings(encoder, dataset):
    print("\n" + "="*60)
    print("Generating Global Embeddings (Batched)")
    print("="*60)
    
    # 1. Prepare data
    all_windows = dataset.windows.astype(np.float32)
    if all_windows.ndim == 2:
        all_windows = all_windows.reshape(len(all_windows), -1, 1)
    
    # 2. Settings for safety
    BATCH_SIZE = 64  # Process 64 windows at a time
    n_samples = len(all_windows)
    
    print(f"Processing {n_samples} windows in batches of {BATCH_SIZE}...")
    
    global_embeddings_list = []
    
    # 3. Manual Batch Loop (Prevents System Crash)
    # We use tqdm to show a progress bar
    for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="Encoding"):
        # Select batch
        batch_data = all_windows[i : i + BATCH_SIZE]
        
        # Encode ONLY this batch
        # Result shape: (BATCH_SIZE, 2000, 128)
        batch_timestamp_emb = encoder.encode(batch_data, batch_size=BATCH_SIZE)

        # Pool IMMEDIATELY (Shrink data before storing)
        # Using Max Pooling as recommended for classification
        mean_pool = batch_timestamp_emb.mean(axis=1)
        max_pool = batch_timestamp_emb.max(axis=1)
        batch_global = np.concatenate([mean_pool, max_pool], axis=1)
        
        # Store only the small pooled vectors
        global_embeddings_list.append(batch_global)
        
        # Explicitly delete large temp array to free memory
        del batch_timestamp_emb
        
    # 4. Combine all small batches
    global_embeddings = np.concatenate(global_embeddings_list, axis=0)
    
    print(f"Generated embeddings shape: {global_embeddings.shape}")
    
    # Collect metadata
    metadata = []
    for idx in range(len(dataset)):
        metadata.append({
            "file": dataset.file_map[idx],
            "directory": dataset.dir_map[idx],
            "idx": idx,
            "label": int(dataset.labels[idx])
        })
    
    return global_embeddings, dataset.labels, metadata
# ====================================================
# 5. SAVE RESULTS
# ====================================================

def save_results(embeddings, labels, metadata):
    # Ensure embeddings directory exists
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Save embeddings and labels
    np.save(os.path.join(EMBEDDINGS_DIR, "pcg_embeddings.npy"), embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, "pcg_labels.npy"), labels)
    
    # Save metadata
    with open(os.path.join(EMBEDDINGS_DIR, "embedding_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    # Save a summary file
    summary_path = os.path.join(EMBEDDINGS_DIR, "embedding_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PCG Embedding Generation Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Total embeddings: {len(embeddings)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n")
        f.write(f"\nLabel distribution:\n")
        
        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: "Normal", 1: "Abnormal", -1: "Unknown"}
        for val, count in zip(unique, counts):
            name = label_names.get(val, f"Unknown({val})")
            percentage = count/len(labels)*100
            f.write(f"  {name}: {count} ({percentage:.1f}%)\n")
    
    print("\n✅ Saved all files:")
    print(f"  1. pcg_embeddings.npy - {embeddings.shape}")
    print(f"  2. pcg_labels.npy - {labels.shape}")
    print(f"  3. embedding_metadata.pkl - {len(metadata)} entries")
    print(f"  4. embedding_summary.txt - Summary statistics")
    
    # Print quick stats
    print(f"\n📊 Quick stats:")
    print(f"  Embedding mean: {np.mean(embeddings):.4f} ± {np.std(embeddings):.4f}")
    print(f"  Min/Max: [{np.min(embeddings):.4f}, {np.max(embeddings):.4f}]")

# ====================================================
# 6. LOAD PRE-TRAINED MODEL (UTILITY FUNCTION)
# ====================================================

def load_trained_model(model_path=None):
    """
    Load a previously trained TS2Vec model.
    Useful if you want to skip training and use existing model.
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "ts2vec_pcg_model")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    print(f"Loading pre-trained model from {model_path}")
    encoder = TS2Vec(input_dims=1, output_dims=128, device=str(DEVICE))
    encoder.load(model_path)
    
    return encoder

# ====================================================
# MAIN PIPELINE
# ====================================================

def main():
    
   
    # Load your preprocessed data
    windows, labels, fmap, dmap = load_preprocessed_data()
    
    # Create dataset
    dataset = PCGDataset(windows, labels, fmap, dmap)
    
    # Check if model already exists
    model_path = os.path.join(MODELS_DIR, "ts2vec_pcg_model")
    if os.path.exists(model_path):
        response = input("\nPre-trained model found. Retrain? (y/n, default=n): ").strip().lower()
        if response in ['y', 'yes']:
            encoder = train_ts2vec(dataset)
        else:
            encoder = load_trained_model(model_path)
    else:
        encoder = train_ts2vec(dataset)
    
    # Generate embeddings
    raw = encoder.encode(dataset.windows[:10].astype(np.float32).reshape(10, -1, 1))
    print(raw.min(), raw.max(), raw.mean(), raw.std())
    embeddings, labels, metadata = generate_embeddings(encoder, dataset)
    
    # Save results
    save_results(embeddings, labels, metadata)
    
    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE!")
    print("="*60)
    
if __name__ == "__main__":
    main()