"""
PCG Embeddings Dataloader for Downstream Tasks
Supports classification, clustering, and visualization
"""

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Optional, Tuple, Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

def extract_from_dataloader(
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    verbose: bool = True,
    return_tensors: bool = False
) -> Tuple[Union[np.ndarray, torch.Tensor], 
           Union[np.ndarray, torch.Tensor], 
           List[Dict]]:
    """
    Extract all embeddings, labels, and metadata from a DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader yielding (embeddings, labels, metadata_list)
        max_batches: Maximum number of batches to process (None for all)
        verbose: Whether to print progress information
        return_tensors: If True, return torch tensors instead of numpy arrays
    
    Returns:
        Tuple containing:
            - embeddings: Array/tensor of shape (n_samples, embedding_dim)
            - labels: Array/tensor of shape (n_samples,)
            - metadata: List of dicts with length n_samples
    
    Example:
        >>> from pcg_embedding_dataloader import PCGEmbeddingDataset
        >>> from torch.utils.data import DataLoader
        >>> 
        >>> # Create dataloader
        >>> dataset = PCGEmbeddingDataset()
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=False)
        >>> 
        >>> # Extract all data
        >>> X, y, metadata = extract_from_dataloader(loader)
        >>> print(f"Extracted {X.shape[0]} samples with {X.shape[1]} dimensions")
        >>> print(f"Label distribution: normal={sum(y==0)}, abnormal={sum(y==1)}")
    """
    
    # Initialize containers
    all_embeddings = []
    all_labels = []
    all_metadata = []
    
    # Track if we're using torch tensors
    using_torch = False
    
    # Process batches
    total_batches = len(dataloader) if hasattr(dataloader, '__len__') else 'unknown'
    
    for batch_idx, batch_data in enumerate(dataloader):
        # Stop if we've reached max_batches
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # Unpack batch (handles different return formats)
        if len(batch_data) == 3:
            embeddings, labels, metadata_list = batch_data
        else:
            raise ValueError(f"Expected 3 items per batch, got {len(batch_data)}")
        
        # Check if we're dealing with torch tensors
        if isinstance(embeddings, torch.Tensor):
            using_torch = True
        
        # Store batch data
        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_metadata.extend(metadata_list)
        
        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Processed batch {batch_idx + 1}/{total_batches}")
    
    # Concatenate all batches
    if using_torch and not return_tensors:
        # Convert torch to numpy
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
    elif using_torch and return_tensors:
        # Keep as torch tensors
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
    else:
        # Already numpy arrays
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
    
    if verbose:
        print(f"\n✅ Extraction complete!")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Metadata entries: {len(all_metadata)}")
        
        # Print label distribution if numpy
        if isinstance(labels, np.ndarray):
            unique, counts = np.unique(labels, return_counts=True)
            label_dist = ", ".join([f"{int(u)}: {c}" for u, c in zip(unique, counts)])
            print(f"   Label distribution: {label_dist}")
    
    return embeddings, labels, all_metadata




class PCGEmbeddingDataset(Dataset):
    """
    Dataset class for PCG embeddings with multiple access patterns.
    
    Features:
    - Load embeddings, labels, and metadata
    - Filter by class (normal/abnormal/unknown)
    - Filter by source directory
    - Get embeddings by original filename
    - Balanced sampling for imbalanced classes
    """
    
    def __init__(self, 
                 embeddings_dir: str = "embeddings/",
                 transform=None,
                 target_transform=None,
                 include_unknown: bool = False,
                 directories: Optional[List[str]] = None):
        """
        Args:
            embeddings_dir: Directory containing embedding files
            transform: Optional transform to apply to embeddings
            target_transform: Optional transform to apply to labels
            include_unknown: Whether to include unknown (-1) labels
            directories: Optional list of directories to filter by
        """
        self.embeddings_dir = embeddings_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Load all data
        self.embeddings = np.load(os.path.join(embeddings_dir, "pcg_embeddings.npy"))
        self.labels = np.load(os.path.join(embeddings_dir, "pcg_labels.npy"))
        
        with open(os.path.join(embeddings_dir, "embedding_metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        
        # Create filename to index mapping
        self.filename_to_idx = {meta['file']: idx for idx, meta in enumerate(self.metadata)}
        
        # Apply filters
        self._apply_filters(include_unknown, directories)
        
        # Class mapping
        self.class_names = {0: "Normal", 1: "Abnormal", -1: "Unknown"}
        self.class_to_idx = {0: 0, 1: 1, -1: 2}
        self.idx_to_class = {0: 0, 1: 1, 2: -1}
        
        print(f"Dataset initialized with {len(self)} samples")
        self._print_class_distribution()
    
    def _apply_filters(self, include_unknown: bool, directories: Optional[List[str]]):
        """Apply filtering based on class and directory"""
        keep_mask = np.ones(len(self.labels), dtype=bool)
        
        # Filter by class
        if not include_unknown:
            keep_mask &= (self.labels != -1)
        
        # Filter by directory
        if directories:
            dir_mask = np.zeros(len(self.labels), dtype=bool)
            for i, meta in enumerate(self.metadata):
                if meta['directory'] in directories:
                    dir_mask[i] = True
            keep_mask &= dir_mask
        
        # Apply mask
        if not np.all(keep_mask):
            self.embeddings = self.embeddings[keep_mask]
            self.labels = self.labels[keep_mask]
            self.metadata = [self.metadata[i] for i in range(len(keep_mask)) if keep_mask[i]]
            
            # Rebuild filename mapping
            self.filename_to_idx = {meta['file']: idx for idx, meta in enumerate(self.metadata)}
    
    def _print_class_distribution(self):
        """Print current class distribution"""
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nClass distribution:")
        for val, count in zip(unique, counts):
            name = self.class_names.get(val, f"Unknown({val})")
            print(f"  {name}: {count} ({count/len(self)*100:.1f}%)")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Returns:
            embedding: numpy array or transformed tensor
            label: integer label
            metadata: dictionary with file info
        """
        embedding = self.embeddings[idx].astype(np.float32)
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return embedding, label, metadata
    
    # ====================================================
    # ACCESS METHODS
    # ====================================================
    
    def get_by_filename(self, filename: str) -> Tuple[np.ndarray, int, dict]:
        """Get embedding by original filename"""
        idx = self.filename_to_idx[filename]
        return self[idx]
    
    def get_by_directory(self, directory: str) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Get all embeddings from a specific directory"""
        indices = [i for i, meta in enumerate(self.metadata) if meta['directory'] == directory]
        return self.embeddings[indices], self.labels[indices], [self.metadata[i] for i in indices]
    
    def get_by_class(self, class_label: int) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Get all embeddings of a specific class"""
        mask = self.labels == class_label
        return self.embeddings[mask], self.labels[mask], [self.metadata[i] for i in np.where(mask)[0]]
    
    def get_indices_by_class(self, class_label: int) -> np.ndarray:
        """Get indices of samples belonging to a class"""
        return np.where(self.labels == class_label)[0]
    
    # ====================================================
    # DATALOADER FACTORY METHODS
    # ====================================================
    
    def get_loader(self, 
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 0,
                   pin_memory: bool = True,
                   drop_last: bool = False) -> DataLoader:
        """Get a standard dataloader for the entire dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def get_train_val_loaders(self,
                             val_size: float = 0.2,
                             batch_size: int = 32,
                             stratify: bool = True,
                             random_seed: int = 42,
                             num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """
        Create stratified train/validation splits
        
        Returns:
            train_loader, val_loader
        """
        indices = np.arange(len(self))
        
        if stratify:
            # Stratified split
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                stratify=self.labels,
                random_state=random_seed
            )
        else:
            # Random split
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=random_seed
            )
        
        train_dataset = Subset(self, train_idx)
        val_dataset = Subset(self, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_kfold_loaders(self,
                         n_splits: int = 5,
                         batch_size: int = 32,
                         random_seed: int = 42,
                         num_workers: int = 0) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create k-fold cross-validation loaders
        
        Returns:
            List of (train_loader, val_loader) for each fold
        """
        # Only use samples with known labels for k-fold
        known_mask = self.labels != -1
        known_indices = np.where(known_mask)[0]
        known_labels = self.labels[known_mask]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        
        fold_loaders = []
        
        for train_idx, val_idx in skf.split(known_indices, known_labels):
            # Map back to original indices
            train_orig_idx = known_indices[train_idx]
            val_orig_idx = known_indices[val_idx]
            
            train_dataset = Subset(self, train_orig_idx)
            val_dataset = Subset(self, val_orig_idx)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            fold_loaders.append((train_loader, val_loader))
        
        return fold_loaders
    
    def get_balanced_loader(self,
        batch_size: int = 32,
        samples_per_class: Optional[int] = None,
        num_workers: int = 0,
        epochs: int = 1) -> DataLoader:
        """
        Create a balanced dataloader by oversampling minority classes
        
        Args:
            batch_size: Batch size
            samples_per_class: Number of samples per class per epoch 
                            (if None, uses minority class size)
            num_workers: Number of workers for dataloader
            epochs: Number of epochs worth of balanced samples to generate
        
        Returns:
            DataLoader that yields balanced batches
        """
        # Get indices for each class (excluding unknown)
        normal_indices = self.get_indices_by_class(0)
        abnormal_indices = self.get_indices_by_class(1)
        
        # Determine minority class size
        minority_size = min(len(normal_indices), len(abnormal_indices))
        majority_size = max(len(normal_indices), len(abnormal_indices))
        
        # Set samples per class
        if samples_per_class is None:
            samples_per_class = minority_size
        
        print(f"Balanced loader: {samples_per_class} samples per class")
        print(f"  Normal: {len(normal_indices)} available → sampling {samples_per_class}")
        print(f"  Abnormal: {len(abnormal_indices)} available → sampling {samples_per_class}")
        
        # Create balanced indices for specified number of epochs
        balanced_indices = []
        
        for epoch in range(epochs):
            # For each class, randomly sample (with replacement if needed)
            if samples_per_class <= len(normal_indices):
                # If we need fewer than available, sample without replacement
                epoch_normal = np.random.choice(normal_indices, samples_per_class, replace=False)
            else:
                # If we need more than available, sample with replacement
                epoch_normal = np.random.choice(normal_indices, samples_per_class, replace=True)
            
            if samples_per_class <= len(abnormal_indices):
                epoch_abnormal = np.random.choice(abnormal_indices, samples_per_class, replace=False)
            else:
                epoch_abnormal = np.random.choice(abnormal_indices, samples_per_class, replace=True)
            
            # Combine and shuffle for this epoch
            epoch_indices = np.concatenate([epoch_normal, epoch_abnormal])
            np.random.shuffle(epoch_indices)
            balanced_indices.extend(epoch_indices)
        
        balanced_indices = np.array(balanced_indices)
        
        # Create dataset and loader
        balanced_dataset = Subset(self, balanced_indices)
        
        return DataLoader(
            balanced_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle each epoch
            num_workers=num_workers,
            pin_memory=True
        )

# ====================================================
# TRANSFORM CLASSES FOR DATA AUGMENTATION
# ====================================================

class ToTensor:
    """Convert numpy array to torch tensor"""
    def __call__(self, x):
        return torch.from_numpy(x)


class NormalizeEmbedding:
    """Normalize embedding to have zero mean and unit variance"""
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + self.eps)
        else:
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + self.eps)


class AddGaussianNoise:
    """Add Gaussian noise to embeddings for augmentation"""
    def __init__(self, std=0.01):
        self.std = std
    
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.std
            return x + noise
        else:
            noise = np.random.randn(*x.shape) * self.std
            return x + noise


# ====================================================
# EXAMPLE USAGE
# ====================================================

if __name__ == "__main__":
    print("="*60)
    print("PCG Embeddings Dataloader - Example Usage")
    print("="*60)
    
    # 1. Basic dataset
    print("\n1. Basic dataset with all samples:")
    dataset = PCGEmbeddingDataset(include_unknown=False)
    print(f"   Total samples: {len(dataset)}")
    
    # Get a single sample
    embedding, label, metadata = dataset[0]
    print(f"   Sample shape: {embedding.shape}")
    print(f"   Label: {label} ({dataset.class_names[label]})")
    print(f"   Source: {metadata['file']}")
    
    # 2. Create simple dataloader
    print("\n2. Simple dataloader:")
    loader = dataset.get_loader(batch_size=32, shuffle=True)
    for batch_idx, (embeddings, labels, metadata_list) in enumerate(loader):
        print(f"   Batch {batch_idx}: embeddings {embeddings.shape}, labels {labels.shape}")
        if batch_idx == 0:  # Just show first batch
            break
    
    # 3. Train/validation split
    print("\n3. Train/validation split:")
    train_loader, val_loader = dataset.get_train_val_loaders(val_size=0.2, batch_size=16)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # 4. Access by class
    print("\n4. Access by class:")
    normal_embeddings, normal_labels, normal_metadata = dataset.get_by_class(0)
    print(f"   Normal samples: {len(normal_embeddings)}")
    
    abnormal_embeddings, abnormal_labels, abnormal_metadata = dataset.get_by_class(1)
    print(f"   Abnormal samples: {len(abnormal_embeddings)}")
    
    # 5. K-fold cross-validation
    print("\n5. 5-fold cross-validation:")
    fold_loaders = dataset.get_kfold_loaders(n_splits=5, batch_size=16)
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"   Fold {fold+1}: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # 6. Balanced loader (for imbalanced datasets)
    print("\n6. Balanced loader (oversamples minority class):")
    balanced_loader = dataset.get_balanced_loader(batch_size=16, samples_per_class=100)
    print(f"   Batches per epoch: {len(balanced_loader)}")
    
    # 7. With transforms
    print("\n7. Dataset with transforms:")
    transformed_dataset = PCGEmbeddingDataset(
        transform=NormalizeEmbedding(),
        target_transform=lambda x: torch.tensor(x, dtype=torch.long)
    )
    embedding, label, _ = transformed_dataset[0]
    print(f"   Transformed embedding - mean: {embedding.mean():.4f}, std: {embedding.std():.4f}")
    print(f"   Transformed label type: {type(label)}")
    
    # 8. Filter by directory
    print("\n8. Filter by directory (training-a only):")
    dir_dataset = PCGEmbeddingDataset(directories=['training-a'])
    print(f"   Samples from training-a: {len(dir_dataset)}")
    
    print("\n✅ Dataloader examples complete!")