import os
import sys
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

# ====================================================
# SAMPLING RATE SELECTION MENU
# ====================================================

def get_sampling_rate_choice():
    """
    Interactive menu to choose sampling rate.
    Returns: (target_sr, folder_name)
    """
    print("\n" + "="*60)
    print("SAMPLING RATE SELECTION")
    print("="*60)
    print("\nChoose sampling rate for preprocessing:")
    print("  1. 1000 Hz (1 kHz) - Recommended for heart sounds")
    print("  2. 2000 Hz (2 kHz) - Original sampling rate")
    print("  3. 4000 Hz (4 kHz) - Higher resolution")
    print("  4. 8000 Hz (8 kHz) - For compatibility with some models")
    print("  5. 16000 Hz (16 kHz) - For wav2vec/Whisper compatibility")
    print("  6. Custom (enter your own)")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    sampling_rates = {
        '1': 1000,
        '2': 2000,
        '3': 4000,
        '4': 8000,
        '5': 16000
    }
    
    if choice in sampling_rates:
        target_sr = sampling_rates[choice]
        folder_name = f"processed_{target_sr}hz"
    elif choice == '6':
        target_sr = int(input("Enter custom sampling rate (Hz): ").strip())
        folder_name = f"processed_{target_sr}hz"
    else:
        print("Invalid choice. Using default 1000 Hz.")
        target_sr = 1000
        folder_name = "processed_1000hz"
    
    # Window size: 2 seconds at chosen sampling rate
    win_size_seconds = 2
    win_size = win_size_seconds * target_sr
    hop_size = win_size // 2  # 50% overlap
    
    print(f"\n✅ Selected configuration:")
    print(f"   Sampling rate: {target_sr} Hz")
    print(f"   Window size: {win_size_seconds} seconds ({win_size} samples)")
    print(f"   Hop size: {hop_size} samples (50% overlap)")
    print(f"   Output folder: {folder_name}")
    
    return target_sr, win_size, hop_size, folder_name

# ====================================================
# CREATE OUTPUT DIRECTORY
# ====================================================

def create_output_directory(base_processed_dir, folder_name):
    """
    Create a unique output directory for this sampling rate.
    If directory exists, ask user what to do.
    """
    output_dir = os.path.join(base_processed_dir, folder_name)
    
    if os.path.exists(output_dir):
        print(f"\n⚠️  Directory already exists: {output_dir}")
        print("Options:")
        print("  1. Overwrite (delete existing files)")
        print("  2. Append (add to existing, may cause duplicates)")
        print("  3. Create new version (add _v2, _v3, etc.)")
        print("  4. Cancel")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            import shutil
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"✅ Existing directory removed and recreated.")
        elif choice == '2':
            print(f"✅ Will append to existing directory.")
        elif choice == '3':
            version = 2
            while True:
                new_folder = f"{folder_name}_v{version}"
                new_output_dir = os.path.join(base_processed_dir, new_folder)
                if not os.path.exists(new_output_dir):
                    output_dir = new_output_dir
                    folder_name = new_folder
                    os.makedirs(output_dir)
                    print(f"✅ Created new directory: {folder_name}")
                    break
                version += 1
        else:
            print("❌ Cancelled.")
            return None, None
    else:
        os.makedirs(output_dir)
        print(f"✅ Created directory: {output_dir}")
    
    return output_dir, folder_name

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
# 2. AUDIO PROCESSING FUNCTIONS (MODIFIED)
# ====================================================
def bandpass_filter(x, sr, lowcut=25, highcut=400):
    """
    Bandpass filter optimized for heart sounds (25-400Hz).
    Now accepts sampling rate as parameter.
    """
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
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


def slice_windows(x, label, win_size, hop_size):
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
# 3. MAIN PREPROCESSING FUNCTION (MODIFIED)
# ====================================================
def preprocess_dataset(output_dir, target_sr, win_size, hop_size, folder_name):
    """
    Main preprocessing pipeline using REFERENCE.csv files for labels.
    Now accepts output directory and sampling parameters.
    """
    print("=" * 60)
    print("PhysioNet 2016 Heart Sound Preprocessing")
    print(f"Target Sampling Rate: {target_sr} Hz")
    print(f"Output Directory: {folder_name}")
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
        'unknown_label_windows': 0,
        'target_sr': target_sr,
        'win_size': win_size,
        'hop_size': hop_size
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
            if sr != target_sr:
                x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
            
            # Apply bandpass filter (adjust cutoff based on sampling rate)
            max_freq = target_sr / 2
            highcut = min(400, max_freq - 1)  # Don't exceed Nyquist
            x = bandpass_filter(x, target_sr, lowcut=25, highcut=highcut)
            
            # Slice into windows
            windows, window_labels = slice_windows(x, label, win_size, hop_size)
            
            if len(windows) == 0:
                stats['skipped_files'] += 1
                continue
            
            # Normalize each window
            windows = np.array([normalize_robust(w) for w in windows])
            
            # Save windows and labels
            base_name = os.path.splitext(filename)[0]
            
            # Create unique identifier with directory prefix
            unique_id = f"{directory}_{base_name}"
            
            windows_file = os.path.join(output_dir, f"{unique_id}_windows.npy")
            labels_file = os.path.join(output_dir, f"{unique_id}_labels.npy")
            
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
    Sampling Rate: {target_sr} Hz
    Window Size: {win_size} samples ({win_size/target_sr} seconds)
    Hop Size: {hop_size} samples ({hop_size/target_sr} seconds)
    
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
    stats_file = os.path.join(output_dir, "preprocessing_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("PhysioNet 2016 Preprocessing Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(summary_text)
    
    print(f"\nPreprocessed data saved to: {output_dir}")
    print(f"Statistics saved to: {stats_file}")
    
    # Create combined dataset
    create_combined_dataset(output_dir)
    
    return stats


def create_combined_dataset(output_dir):
    """
    Create a combined dataset file with all windows and labels.
    Now accepts output directory as parameter.
    """
    print("\nCreating combined dataset file...")
    
    all_windows = []
    all_labels = []
    file_mapping = []
    directory_mapping = []
    
    # Find all window files
    window_files = [f for f in os.listdir(output_dir) 
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
        
        windows_path = os.path.join(output_dir, wf)
        labels_path = os.path.join(output_dir, f"{unique_id}_labels.npy")
        
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
        np.save(os.path.join(output_dir, "combined_windows.npy"), combined_windows)
        np.save(os.path.join(output_dir, "combined_labels.npy"), combined_labels)
        np.save(os.path.join(output_dir, "combined_file_mapping.npy"), np.array(file_mapping))
        np.save(os.path.join(output_dir, "combined_directory_mapping.npy"), np.array(directory_mapping))
        
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
        class_dist_file = os.path.join(output_dir, "class_distribution.txt")
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
# 4. DATA LOADER FOR TRAINING (MODIFIED)
# ====================================================
def load_preprocessed_data(sampling_rate=None):
    """
    Load the preprocessed data for training.
    If sampling_rate is None, shows menu to choose which dataset to load.
    Returns: (windows, labels, file_mapping, metadata)
    """
    base_processed_dir = "../data/processed/"
    
    # If sampling rate not specified, show available datasets
    if sampling_rate is None:
        # List available processed datasets
        available = []
        for item in os.listdir(base_processed_dir):
            if item.startswith('processed_') and os.path.isdir(os.path.join(base_processed_dir, item)):
                available.append(item)
        
        if not available:
            print("No preprocessed datasets found. Run preprocessing first.")
            return None, None, None, None
        
        print("\n📂 Available preprocessed datasets:")
        for i, dataset in enumerate(available):
            # Extract sampling rate from folder name
            sr = dataset.replace('processed_', '').replace('hz', '')
            print(f"  {i+1}. {dataset} ({sr} Hz)")
        
        choice = input("\nSelect dataset (enter number): ").strip()
        try:
            idx = int(choice) - 1
            folder_name = available[idx]
        except:
            print("Invalid choice. Using first dataset.")
            folder_name = available[0]
        
        data_dir = os.path.join(base_processed_dir, folder_name)
    else:
        folder_name = f"processed_{sampling_rate}hz"
        data_dir = os.path.join(base_processed_dir, folder_name)
    
    windows_path = os.path.join(data_dir, "combined_windows.npy")
    labels_path = os.path.join(data_dir, "combined_labels.npy")
    mapping_path = os.path.join(data_dir, "combined_file_mapping.npy")
    dir_mapping_path = os.path.join(data_dir, "combined_directory_mapping.npy")
    
    if not os.path.exists(windows_path):
        print(f"Preprocessed data not found at {data_dir}. Run preprocessing first.")
        return None, None, None, None
    
    windows = np.load(windows_path)
    labels = np.load(labels_path)
    file_mapping = np.load(mapping_path) if os.path.exists(mapping_path) else None
    dir_mapping = np.load(dir_mapping_path) if os.path.exists(dir_mapping_path) else None
    
    # Extract sampling rate from folder name for info
    sr = folder_name.replace('processed_', '').replace('hz', '')
    
    print(f"✅ Loaded preprocessed data from {folder_name}:")
    print(f"   Sampling rate: {sr} Hz")
    print(f"   Windows: {windows.shape}")
    print(f"   Labels: {labels.shape}")
    
    metadata = {
        'sampling_rate': int(sr),
        'folder_name': folder_name,
        'data_dir': data_dir
    }
    
    return windows, labels, file_mapping, metadata


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
# 6. UTILITY: LIST ALL PROCESSED DATASETS
# ====================================================
def list_processed_datasets():
    """
    List all available preprocessed datasets with their statistics.
    """
    base_processed_dir = "../data/processed/"
    
    if not os.path.exists(base_processed_dir):
        print("No processed datasets found.")
        return
    
    print("\n" + "="*60)
    print("AVAILABLE PREPROCESSED DATASETS")
    print("="*60)
    
    for item in os.listdir(base_processed_dir):
        if item.startswith('processed_') and os.path.isdir(os.path.join(base_processed_dir, item)):
            data_dir = os.path.join(base_processed_dir, item)
            stats_file = os.path.join(data_dir, "preprocessing_stats.txt")
            
            # Extract sampling rate
            sr = item.replace('processed_', '').replace('hz', '')
            
            # Try to load stats
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    content = f.read()
                    # Extract window count
                    import re
                    match = re.search(r'Total windows generated: (\d+)', content)
                    windows_count = match.group(1) if match else "unknown"
            else:
                windows_count = "unknown"
            
            print(f"\n  📁 {item}")
            print(f"     Sampling Rate: {sr} Hz")
            print(f"     Windows: {windows_count}")
            print(f"     Path: {data_dir}")


# ====================================================
# MAIN EXECUTION
# ====================================================
# ====================================================
# MAIN EXECUTION
# ====================================================
if __name__ == "__main__":
    # Check for auto-mode via environment variable
    AUTO_MODE_SR = os.environ.get('TARGET_SR')
    
    if AUTO_MODE_SR is not None:
        # AUTO-MODE: Run without interactive menus
        print(f"\n🤖 Auto-mode: TARGET_SR={AUTO_MODE_SR} Hz")
        target_sr = int(AUTO_MODE_SR)
        win_size = target_sr * 2
        hop_size = win_size // 2
        folder_name = f"processed_{target_sr}hz"
        base_processed_dir = "../data/processed/"
        output_dir = os.path.join(base_processed_dir, folder_name)
        
        # Handle existing directory
        if os.path.exists(output_dir):
            version = 2
            while True:
                new_folder = f"{folder_name}_v{version}"
                new_output_dir = os.path.join(base_processed_dir, new_folder)
                if not os.path.exists(new_output_dir):
                    output_dir = new_output_dir
                    folder_name = new_folder
                    break
                version += 1
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run preprocessing
        stats = preprocess_dataset(output_dir, target_sr, win_size, hop_size, folder_name)
        sys.exit(0)
    
    # INTERACTIVE MODE: Original code
    print("PhysioNet 2016 PCG Preprocessing Pipeline")
    print("Interactive Sampling Rate Selection")
    print("=" * 60)
    
    check_dataset_structure()
    
    print("\n" + "="*60)
    print("MAIN MENU")
    print("="*60)
    print("\nWhat would you like to do?")
    print("  1. Preprocess new dataset (choose sampling rate)")
    print("  2. List existing preprocessed datasets")
    print("  3. Load existing dataset and show example")
    print("  4. Exit")
    
    action = input("\nEnter choice (1-4): ").strip()
    
    if action == '1':
        target_sr, win_size, hop_size, folder_name = get_sampling_rate_choice()
        base_processed_dir = "../data/processed/"
        output_dir, folder_name = create_output_directory(base_processed_dir, folder_name)
        
        if output_dir:
            stats = preprocess_dataset(output_dir, target_sr, win_size, hop_size, folder_name)
            
            if stats and stats['processed_files'] > 0:
                print("\n" + "=" * 60)
                print("EXAMPLE: Loading the preprocessed data")
                print("=" * 60)
                windows, labels, mapping, metadata = load_preprocessed_data(target_sr)
                
                if windows is not None:
                    print(f"\n✅ Successfully preprocessed {folder_name}")
                    print(f"   Data shape: {windows.shape}")
                    print(f"   Labels shape: {labels.shape}")
                    print(f"   Ready for TS2Vec: reshape to [-1, {win_size}, 1]")
    
    elif action == '2':
        list_processed_datasets()
    
    elif action == '3':
        windows, labels, mapping, metadata = load_preprocessed_data()
        if windows is not None:
            print("\n📝 Example usage in your training script:")
            print(""" ... """)
    
    else:
        print("Exiting...")