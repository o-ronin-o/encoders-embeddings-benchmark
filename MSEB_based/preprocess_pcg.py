import os
import numpy as np
import librosa
import scipy.signal as signal
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ====================================================
# CONFIGURATION FOR PHYSIONET 2016 DATASET
# ====================================================

RAW_DATA = "../data/raw/"               # Directory with Physionet data (with subdirectories)
PROCESSED_DATA = "../data/processed/"   # Where we save preprocessed windows

os.makedirs(PROCESSED_DATA, exist_ok=True)

# Dataset-specific parameters
ORIGINAL_SR = 2000                   # All files are resampled to 2000Hz
TARGET_SR = 1000                     # Downsample to 1kHz for processing
WIN_SIZE_SECONDS = 2                 # 2-second windows
WIN_SIZE = WIN_SIZE_SECONDS * TARGET_SR  # 2000 samples at 1kHz
HOP_SIZE = WIN_SIZE // 2             # 50% overlap (1000 samples)

# Filter parameters for heart sounds
LOWCUT = 25                          # Hz - removes breathing noise
HIGHCUT = 400                        # Hz - heart sound frequencies

# Quality control thresholds
MIN_WINDOW_LENGTH = 1.0 * TARGET_SR  # Minimum 1-second window to keep


# ====================================================
# 1. LOAD LABELS FROM REFERENCE.CSV FILES
# ====================================================
def load_labels_from_csv(raw_data_dir):
    """
    Load labels from REFERENCE.csv files in each training directory.
    
    REFERENCE.csv format (example):
    recording,label
    a0001,normal
    a0002,abnormal
    or
    a0001 1
    a0002 -1
    
    Returns: dictionary mapping filename -> label (0=normal, 1=abnormal, -1=unknown)
    """
    print("Loading labels from REFERENCE.csv files...")
    labels_dict = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower() == 'reference.csv':
                csv_path = os.path.join(root, file)
                directory_name = os.path.basename(root)
                print(f"  Reading {directory_name}/REFERENCE.csv")
                
                try:
                    # Try different CSV formats
                    try:
                        # Try standard CSV with header
                        df = pd.read_csv(csv_path)
                    except:
                        # Try CSV without header or with different separator
                        df = pd.read_csv(csv_path, header=None, sep=',')
                    
                    # Determine which columns contain filename and label
                    if len(df.columns) >= 2:
                        # Check if first row contains header
                        first_val = str(df.iloc[0, 0]).lower()
                        if first_val in ['recording', 'filename', 'file']:
                            # Has header, skip first row for data
                            data_start = 1
                        else:
                            # No header
                            data_start = 0
                        
                        for i in range(data_start, len(df)):
                            # Get filename and label
                            filename = str(df.iloc[i, 0]).strip()
                            label_str = str(df.iloc[i, 1]).strip().lower()
                            
                            # Remove .wav extension if present
                            if filename.endswith('.wav'):
                                filename = filename[:-4]
                            
                            # Convert label
                            if label_str in ['normal', 'n', '1']:
                                label = 0
                            elif label_str in ['abnormal', 'a', '-1']:
                                label = 1
                            else:
                                # Try numeric conversion
                                try:
                                    label_val = int(label_str)
                                    if label_val == 1:
                                        label = 0  # Normal
                                    elif label_val == -1:
                                        label = 1  # Abnormal
                                    else:
                                        label = -1  # Unknown
                                except:
                                    label = -1  # Unknown
                            
                            # Map to .wav file
                            wav_filename = filename + '.wav'
                            labels_dict[wav_filename] = label
                            
                except Exception as e:
                    print(f"    Error reading {csv_path}: {e}")
    
    # Also check for a global REFERENCE.csv in the root
    global_ref = os.path.join(raw_data_dir, "REFERENCE.csv")
    if os.path.exists(global_ref) and not labels_dict:
        print("  Reading global REFERENCE.csv")
        try:
            df = pd.read_csv(global_ref)
            if len(df.columns) >= 2:
                for i in range(len(df)):
                    filename = str(df.iloc[i, 0]).strip()
                    if filename.endswith('.wav'):
                        filename = filename[:-4]
                    
                    label_str = str(df.iloc[i, 1]).strip().lower()
                    if label_str in ['normal', 'n', '1']:
                        label = 0
                    elif label_str in ['abnormal', 'a', '-1']:
                        label = 1
                    else:
                        label = -1
                    
                    wav_filename = filename + '.wav'
                    labels_dict[wav_filename] = label
        except Exception as e:
            print(f"    Error reading global REFERENCE.csv: {e}")
    
    print(f"\nLoaded labels for {len(labels_dict)} files")
    
    # Print label distribution
    normal_count = sum(1 for v in labels_dict.values() if v == 0)
    abnormal_count = sum(1 for v in labels_dict.values() if v == 1)
    unknown_count = sum(1 for v in labels_dict.values() if v == -1)
    
    print(f"\nLabel distribution:")
    print(f"  Normal: {normal_count} ({normal_count/len(labels_dict)*100:.1f}%)")
    print(f"  Abnormal: {abnormal_count} ({abnormal_count/len(labels_dict)*100:.1f}%)")
    if unknown_count > 0:
        print(f"  Unknown: {unknown_count} ({unknown_count/len(labels_dict)*100:.1f}%)")
    
    # Show a few examples
    if labels_dict:
        print("\nSample of loaded labels (first 5):")
        sample_items = list(labels_dict.items())[:5]
        for filename, label in sample_items:
            label_text = "Normal" if label == 0 else "Abnormal" if label == 1 else "Unknown"
            print(f"  {filename}: {label_text}")
    
    return labels_dict


# ====================================================
# 2. AUDIO PROCESSING FUNCTIONS
# ====================================================
def bandpass_filter(x, sr=TARGET_SR):
    """
    Bandpass filter optimized for heart sounds (25-400Hz).
    """
    nyquist = sr / 2
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    
    # 3rd order Butterworth filter
    b, a = signal.butter(3, [low, high], btype='band')
    
    # Zero-phase filtering
    return signal.filtfilt(b, a, x)


def normalize_robust(x):
    """
    Z-score normalization with outlier protection.
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))  # Median Absolute Deviation
    
    if mad < 1e-8:
        return x - median
    else:
        return (x - median) / (mad + 1e-8)


def slice_windows(x, label, win_size=WIN_SIZE, hop_size=HOP_SIZE):
    """
    Slice audio into windows.
    Returns: (windows_array, labels_array)
    """
    windows = []
    window_labels = []
    
    if len(x) < win_size:
        # If file is too short, pad it
        x_padded = np.pad(x, (0, max(0, win_size - len(x))), mode='constant')
        windows.append(x_padded)
        window_labels.append(label)
        return np.array(windows), np.array(window_labels)
    
    for start in range(0, len(x) - win_size + 1, hop_size):
        w = x[start:start + win_size]
        windows.append(w)
        window_labels.append(label)
    
    return np.array(windows), np.array(window_labels)


# ====================================================
# 3. MAIN PREPROCESSING FUNCTION
# ====================================================
def preprocess_dataset():
    """
    Main preprocessing pipeline using REFERENCE.csv files for labels.
    """
    print("=" * 60)
    print("PhysioNet 2016 Heart Sound Preprocessing")
    print("Using REFERENCE.csv files for labels")
    print("=" * 60)
    
    # Step 1: Load labels from REFERENCE.csv files
    labels_dict = load_labels_from_csv(RAW_DATA)
    
    if not labels_dict:
        print("\n⚠️  WARNING: No labels loaded from REFERENCE.csv files!")
        print("   Check if REFERENCE.csv files exist in each training directory.")
        response = input("   Continue without labels? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Preprocessing cancelled.")
            return
    
    # Statistics tracking
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'skipped_files': 0,
        'total_windows': 0,
        'normal_windows': 0,
        'abnormal_windows': 0,
        'unknown_label_windows': 0
    }
    
    # Get all .wav files
    wav_files = []
    for root, dirs, files in os.walk(RAW_DATA):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append({
                    'path': os.path.join(root, file),
                    'filename': file,
                    'directory': os.path.basename(root)
                })
    
    stats['total_files'] = len(wav_files)
    print(f"\nFound {stats['total_files']} .wav files")
    
    if stats['total_files'] == 0:
        print(f"No .wav files found in {RAW_DATA}")
        return
    
    # Process each file
    print(f"\nProcessing files...")
    progress_bar = tqdm(wav_files, desc="Preprocessing", unit="file")
    
    for file_info in progress_bar:
        filepath = file_info['path']
        filename = file_info['filename']
        directory = file_info['directory']
        
        try:
            # Get label for this file
            label = labels_dict.get(filename, -1)
            
            # Load audio
            x, sr = librosa.load(filepath, sr=None, mono=True)
            
            # Resample if needed
            if sr != TARGET_SR:
                x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
            
            # Apply bandpass filter
            x = bandpass_filter(x)
            
            # Slice into windows
            windows, window_labels = slice_windows(x, label)
            
            if len(windows) == 0:
                stats['skipped_files'] += 1
                continue
            
            # Normalize each window
            windows = np.array([normalize_robust(w) for w in windows])
            
            # Save windows and labels
            base_name = os.path.splitext(filename)[0]
            
            # Create unique identifier with directory prefix
            unique_id = f"{directory}_{base_name}"
            
            windows_file = os.path.join(PROCESSED_DATA, f"{unique_id}_windows.npy")
            labels_file = os.path.join(PROCESSED_DATA, f"{unique_id}_labels.npy")
            
            np.save(windows_file, windows)
            np.save(labels_file, window_labels)
            
            # Update statistics
            stats['processed_files'] += 1
            stats['total_windows'] += len(windows)
            
            # Count labels
            normal_count = np.sum(window_labels == 0)
            abnormal_count = np.sum(window_labels == 1)
            unknown_count = np.sum(window_labels == -1)
            
            stats['normal_windows'] += normal_count
            stats['abnormal_windows'] += abnormal_count
            stats['unknown_label_windows'] += unknown_count
            
        except Exception as e:
            stats['skipped_files'] += 1
            # Only show first few errors to avoid cluttering output
            if stats['skipped_files'] <= 3:
                print(f"\nError processing {filename}: {str(e)[:100]}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    
    summary_text = f"""
    Total files scanned: {stats['total_files']}
    Successfully processed: {stats['processed_files']}
    Skipped/failed: {stats['skipped_files']}
    
    Total windows generated: {stats['total_windows']}
    
    Label distribution:
      Normal windows: {stats['normal_windows']}
      Abnormal windows: {stats['abnormal_windows']}
      Unknown label: {stats['unknown_label_windows']}
    
    Class ratio (Normal:Abnormal): {stats['normal_windows']}:{stats['abnormal_windows']}
    """
    
    print(summary_text)
    
    # Save statistics
    stats_file = os.path.join(PROCESSED_DATA, "preprocessing_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("PhysioNet 2016 Preprocessing Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(summary_text)
    
    print(f"\nPreprocessed data saved to: {PROCESSED_DATA}")
    print(f"Statistics saved to: {stats_file}")
    
    # Create combined dataset
    create_combined_dataset()
    
    return stats


def create_combined_dataset():
    """
    Create a combined dataset file with all windows and labels.
    """
    print("\nCreating combined dataset file...")
    
    all_windows = []
    all_labels = []
    file_mapping = []
    directory_mapping = []
    
    # Find all window files
    window_files = [f for f in os.listdir(PROCESSED_DATA) 
                   if f.endswith('_windows.npy') and not f.startswith('combined')]
    
    if not window_files:
        print("No window files found to combine.")
        return None, None
    
    print(f"Found {len(window_files)} window files to combine...")
    
    for wf in tqdm(window_files, desc="Combining"):
        # Extract identifier
        unique_id = wf.replace('_windows.npy', '')
        parts = unique_id.split('_', 1)  # Split into directory and filename
        
        if len(parts) >= 2:
            directory = parts[0]
            base_name = parts[1]
        else:
            directory = 'unknown'
            base_name = unique_id
        
        windows_path = os.path.join(PROCESSED_DATA, wf)
        labels_path = os.path.join(PROCESSED_DATA, f"{unique_id}_labels.npy")
        
        if os.path.exists(windows_path) and os.path.exists(labels_path):
            windows = np.load(windows_path)
            labels = np.load(labels_path)
            
            all_windows.append(windows)
            all_labels.append(labels)
            
            # Track which file each window came from
            for i in range(len(windows)):
                file_mapping.append(f"{directory}/{base_name}_window{i}")
                directory_mapping.append(directory)
    
    if all_windows:
        # Concatenate all data
        combined_windows = np.vstack(all_windows)
        combined_labels = np.hstack(all_labels)
        
        # Save combined dataset
        np.save(os.path.join(PROCESSED_DATA, "combined_windows.npy"), combined_windows)
        np.save(os.path.join(PROCESSED_DATA, "combined_labels.npy"), combined_labels)
        np.save(os.path.join(PROCESSED_DATA, "combined_file_mapping.npy"), np.array(file_mapping))
        np.save(os.path.join(PROCESSED_DATA, "combined_directory_mapping.npy"), np.array(directory_mapping))
        
        print(f"\n✅ Combined dataset created:")
        print(f"   Windows shape: {combined_windows.shape}")
        print(f"   Labels shape: {combined_labels.shape}")
        
        # Print class distribution
        normal_mask = combined_labels == 0
        abnormal_mask = combined_labels == 1
        unknown_mask = combined_labels == -1
        
        print(f"   Class distribution:")
        print(f"     Normal: {np.sum(normal_mask)} windows")
        print(f"     Abnormal: {np.sum(abnormal_mask)} windows")
        print(f"     Unknown: {np.sum(unknown_mask)} windows")
        
        # Save class distribution
        class_dist_file = os.path.join(PROCESSED_DATA, "class_distribution.txt")
        with open(class_dist_file, 'w') as f:
            f.write(f"Normal: {np.sum(normal_mask)}\n")
            f.write(f"Abnormal: {np.sum(abnormal_mask)}\n")
            f.write(f"Unknown: {np.sum(unknown_mask)}\n")
            f.write(f"Total: {len(combined_labels)}\n")
        
        return combined_windows, combined_labels
    else:
        print("❌ No data to combine.")
        return None, None


# ====================================================
# 4. DATA LOADER FOR TRAINING
# ====================================================
def load_preprocessed_data():
    """
    Load the preprocessed data for training.
    Returns: (windows, labels, file_mapping)
    """
    windows_path = os.path.join(PROCESSED_DATA, "combined_windows.npy")
    labels_path = os.path.join(PROCESSED_DATA, "combined_labels.npy")
    mapping_path = os.path.join(PROCESSED_DATA, "combined_file_mapping.npy")
    
    if not os.path.exists(windows_path):
        print("Preprocessed data not found. Run preprocessing first.")
        return None, None, None
    
    windows = np.load(windows_path)
    labels = np.load(labels_path)
    file_mapping = np.load(mapping_path) if os.path.exists(mapping_path) else None
    
    print(f"✅ Loaded preprocessed data:")
    print(f"   Windows: {windows.shape} (ready for TS2Vec: reshape to [-1, 2000, 1])")
    print(f"   Labels: {labels.shape}")
    
    return windows, labels, file_mapping


# ====================================================
# 5. CHECK DATASET STRUCTURE
# ====================================================
def check_dataset_structure():
    """
    Check the structure of the raw dataset.
    """
    print("Checking dataset structure...")
    
    file_counts = {
        '.wav': 0,
        '.hea': 0,
        'reference.csv': 0,
        'directories': []
    }
    
    # Walk through directory
    for root, dirs, files in os.walk(RAW_DATA):
        dir_name = os.path.basename(root)
        wav_files = [f for f in files if f.endswith('.wav')]
        hea_files = [f for f in files if f.endswith('.hea')]
        ref_files = [f for f in files if f.lower() == 'reference.csv']
        
        if wav_files:
            file_counts['.wav'] += len(wav_files)
            file_counts['.hea'] += len(hea_files)
            file_counts['reference.csv'] += len(ref_files)
            
            if dir_name not in ['', '.']:
                file_counts['directories'].append(dir_name)
            
            print(f"  {dir_name}:")
            print(f"    .wav files: {len(wav_files)}")
            if ref_files:
                print(f"    REFERENCE.csv: Found ✓")
            else:
                print(f"    REFERENCE.csv: Not found ✗")
    
    print(f"\n📊 Summary:")
    print(f"  Directories with data: {len(file_counts['directories'])}")
    print(f"  Total .wav files: {file_counts['.wav']}")
    print(f"  Total .hea files: {file_counts['.hea']}")
    print(f"  Total REFERENCE.csv files: {file_counts['reference.csv']}")
    
    # List directories
    if file_counts['directories']:
        print(f"\n  Directories found: {', '.join(file_counts['directories'][:5])}")
        if len(file_counts['directories']) > 5:
            print(f"    ... and {len(file_counts['directories']) - 5} more")
    
    return file_counts


# ====================================================
# MAIN EXECUTION
# ====================================================
if __name__ == "__main__":
    print("PhysioNet 2016 PCG Preprocessing Pipeline")
    print("Using REFERENCE.csv files for labels")
    print("=" * 60)
    
    # Check dataset structure
    check_dataset_structure()
    
    # Run preprocessing
    stats = preprocess_dataset()
    
    # Example of how to load the data
    if stats and stats['processed_files'] > 0:
        print("\n" + "=" * 60)
        print("EXAMPLE: Loading data for TS2Vec training")
        print("=" * 60)
        
        windows, labels, mapping = load_preprocessed_data()
        
        if windows is not None:
            print("\n📝 Example usage in your training script:")
            print("""
            from preprocess_pcg import load_preprocessed_data
            
            # Load data
            X, y, _ = load_preprocessed_data()
            
            # Reshape for TS2Vec (samples, timesteps, channels)
            X = X.reshape(-1, 2000, 1)  # 2000 samples per window
            
            # For supervised learning:
            normal_mask = y == 0
            abnormal_mask = y == 1
            
            X_normal = X[normal_mask]
            X_abnormal = X[abnormal_mask]
            
            print(f"Normal samples: {len(X_normal)}")
            print(f"Abnormal samples: {len(X_abnormal)}")
            """)