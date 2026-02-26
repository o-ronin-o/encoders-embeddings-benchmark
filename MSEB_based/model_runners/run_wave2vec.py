"""
Generate embeddings using a Wav2Vec 2.0 / WavLM model.

Creates:
    embeddings/pcg_embeddings_wav2vec.npy
    embeddings/pcg_labels_wav2vec.npy
    embeddings/embedding_metadata_wav2vec.pkl
"""

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# ==================== FILL IN: Import your Wav2Vec model here ====================
# from transformers import Wav2Vec2Model, Wav2Vec2Processor
# ==================================================================================

# ====================================================
# PATHS
# ====================================================
PROCESSED_DATA = "data/processed/"
EMBEDDINGS_DIR = "../embeddings/"
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
# 1. LOAD PREPROCESSED DATA (same as TS2Vec)
# ====================================================
def load_preprocessed_data():
    print("Loading preprocessed PCG windows...")
    windows = np.load(os.path.join(PROCESSED_DATA, "combined_windows.npy"))
    labels = np.load(os.path.join(PROCESSED_DATA, "combined_labels.npy"))
    file_map = np.load(os.path.join(PROCESSED_DATA, "combined_file_mapping.npy"))
    dir_map = np.load(os.path.join(PROCESSED_DATA, "combined_directory_mapping.npy"))

    print(f"Loaded {len(windows)} windows, each shape: {windows.shape[1]} samples")
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "Normal", 1: "Abnormal", -1: "Unknown"}
    for val, count in zip(unique, counts):
        name = label_names.get(val, f"Unknown({val})")
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")
    return windows, labels, file_map, dir_map

# ====================================================
# 2. DATASET WRAPPER (same as TS2Vec)
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
        x = torch.from_numpy(x)            # shape: (2000,)
        # Many Wav2Vec models expect raw audio at 16kHz. You may need to resample.
        # Here we keep the original 1kHz, but you might need to upsample.
        # Option: add a channel dimension if required.
        x = x.unsqueeze(0) if len(x.shape) == 1 else x   # (1, 2000)
        label = int(self.labels[idx])
        meta = {
            "file": self.file_map[idx],
            "directory": self.dir_map[idx],
            "idx": idx
        }
        return x, label, meta

# ====================================================
# 3. INITIALIZE / LOAD WAV2VEC MODEL
# ====================================================
def get_wav2vec_model():
    """
    Load a pre-trained Wav2Vec 2.0 or WavLM model.
    Replace with your actual model loading code.
    """
    print("\n" + "="*60)
    print("Loading Wav2Vec model")
    print("="*60)
    
    # ============== FILL IN: Model name/path ==============
    model_name = "facebook/wav2vec2-base-960h"   # example
    # =======================================================
    
    # ============== FILL IN: HuggingFace loading ===========
    # processor = Wav2Vec2Processor.from_pretrained(model_name)
    # model = Wav2Vec2Model.from_pretrained(model_name)
    # model = model.to(DEVICE)
    # model.eval()
    # return model, processor
    # ========================================================
    
    # Placeholder
    model = None
    processor = None
    return model, processor

# ====================================================
# 4. GENERATE EMBEDDINGS
# ====================================================
def generate_embeddings(model, processor, dataset, batch_size=32):
    print("\n" + "="*60)
    print("Generating Wav2Vec embeddings (batched)")
    print("="*60)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_embeddings = []
    all_labels = []
    all_metadata = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            waveforms, labels, meta_list = batch
            waveforms = waveforms.to(DEVICE)   # shape: (batch, 1, 2000)
            
            # ========= FILL IN: Forward pass & pooling =========
            # If the model expects a specific sampling rate, you may need to resample.
            # Example: outputs = model(waveforms.squeeze(1)).last_hidden_state  # (batch, time, dim)
            # Then pool: mean_pool = outputs.mean(dim=1)
            #           max_pool = outputs.max(dim=1)[0]
            #           emb = torch.cat([mean_pool, max_pool], dim=1)
            # ====================================================
            
            # Placeholder: random embeddings (replace with actual)
            emb = torch.randn(waveforms.size(0), 256).cpu().numpy()
            
            all_embeddings.append(emb)
            all_labels.append(labels.numpy())
            all_metadata.extend(meta_list)
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings, labels, all_metadata

# ====================================================
# 5. SAVE RESULTS (modified filenames)
# ====================================================
def save_results(embeddings, labels, metadata, encoder_name="wav2vec"):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Use encoder-specific filenames
    np.save(os.path.join(EMBEDDINGS_DIR, f"pcg_embeddings_{encoder_name}.npy"), embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, f"pcg_labels_{encoder_name}.npy"), labels)
    
    with open(os.path.join(EMBEDDINGS_DIR, f"embedding_metadata_{encoder_name}.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    summary_path = os.path.join(EMBEDDINGS_DIR, f"embedding_summary_{encoder_name}.txt")
    with open(summary_path, "w") as f:
        f.write(f"PCG Embedding Generation Summary - {encoder_name}\n")
        f.write("="*40 + "\n")
        f.write(f"Total embeddings: {len(embeddings)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n")
        f.write("\nLabel distribution:\n")
        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: "Normal", 1: "Abnormal", -1: "Unknown"}
        for val, count in zip(unique, counts):
            name = label_names.get(val, f"Unknown({val})")
            percentage = count/len(labels)*100
            f.write(f"  {name}: {count} ({percentage:.1f}%)\n")
    
    print(f"\n✅ Saved all files for {encoder_name}:")
    print(f"   embeddings shape: {embeddings.shape}")
    print(f"   labels shape: {labels.shape}")
    print(f"   metadata entries: {len(metadata)}")

# ====================================================
# MAIN
# ====================================================
def main():
    windows, labels, fmap, dmap = load_preprocessed_data()
    dataset = PCGDataset(windows, labels, fmap, dmap)
    
    model, processor = get_wav2vec_model()
    if model is None:
        print("ERROR: Model loading not implemented. Please fill in the code.")
        return
    
    embeddings, labels_out, metadata = generate_embeddings(model, processor, dataset)
    save_results(embeddings, labels_out, metadata, encoder_name="wav2vec")
    
    print("\n" + "="*60)
    print("WAV2VEC EMBEDDING GENERATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()