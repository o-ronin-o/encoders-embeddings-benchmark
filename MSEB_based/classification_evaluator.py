"""
Heart sound classification evaluator adapted for PCG embedding dataloader.

This evaluator works directly with the outputs from PCGEmbeddingDataset:
- embeddings: numpy arrays of shape (N, 256)
- labels: numpy arrays with values 0 (normal), 1 (abnormal), -1 (unknown)
- metadata: list of dicts with file information

Two evaluation strategies:
    'linear_probe'       Logistic regression fitted on training embeddings.
                         Gold standard for embedding quality comparison.
                         Automatically handles train/val/test splits.

    'nearest_centroid'   Classify by cosine distance to per-class mean embeddings.
                         No hyper-parameters to tune. Good for quick validation.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, List, Any, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Label mapping: 0=normal, 1=abnormal, -1=unknown
NORMAL_LABEL = 0
ABNORMAL_LABEL = 1
UNKNOWN_LABEL = -1


class PCGEmbeddingEvaluator:
    """
    Evaluate PCG embeddings on heart sound classification.
    
    This evaluator is designed to work with the PCGEmbeddingDataset outputs:
    - X: embeddings array of shape (n_samples, embedding_dim)
    - y: labels array with values 0 (normal), 1 (abnormal), -1 (unknown)
    
    Key Features:
    - Automatic handling of unknown labels (-1)
    - Multiple evaluation strategies
    - Built-in train/validation/test splitting
    - Cross-validation support
    - Comprehensive metrics reporting
    """
    
    def __init__(
        self,
        strategy: str = "linear_probe",
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
    ) -> None:
        """
        Args:
            strategy: 'linear_probe' or 'nearest_centroid'
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing (if no separate test set)
            val_size: Proportion of training data to use for validation (optional)
        """
        if strategy not in ("linear_probe", "nearest_centroid"):
            raise ValueError(
                f"strategy must be 'linear_probe' or 'nearest_centroid', got: {strategy!r}"
            )
        
        self.strategy = strategy
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        
        # These will be set during evaluation
        self.label_encoder = None
        self.scaler = None
        self.classifier = None
        
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        exclude_unknown: bool = True,
        return_classifier: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate embedding quality for heart sound classification.
        
        Args:
            X: Embeddings array of shape (n_samples, embedding_dim)
            y: Labels array with values 0 (normal), 1 (abnormal), -1 (unknown)
            metadata: Optional list of metadata dicts from dataloader
            X_train, y_train: Optional pre-split training data
            X_val, y_val: Optional pre-split validation data
            X_test, y_test: Optional pre-split test data
            exclude_unknown: If True, exclude unknown (-1) labels from evaluation
            return_classifier: If True, return fitted classifier in results
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        _check_sklearn()
        
        # Filter out unknown labels if requested
        if exclude_unknown:
            known_mask = y != UNKNOWN_LABEL
            X = X[known_mask]
            y = y[known_mask]
            if metadata:
                metadata = [m for i, m in enumerate(metadata) if known_mask[i]]
            
            logger.info(f"Excluded unknown labels. Remaining: {len(y)} samples")
        
        # Handle data splitting
        if X_test is not None and y_test is not None:
            # Use provided test set
            if X_train is None or y_train is None:
                raise ValueError("If test set is provided, training set must also be provided")
            
            X_tr, y_tr = X_train, y_train
            X_te, y_te = X_test, y_test
            
            # Handle validation set if provided
            if X_val is not None and y_val is not None:
                X_tr_combined = np.vstack([X_tr, X_val])
                y_tr_combined = np.concatenate([y_tr, y_val])
                logger.info(f"Using provided splits: train={len(X_tr)} + val={len(X_val)} = {len(X_tr_combined)}, test={len(X_te)}")
            else:
                X_tr_combined = X_tr
                y_tr_combined = y_tr
                logger.info(f"Using provided splits: train={len(X_tr)}, test={len(X_te)}")
                
        else:
            # Split the data
            X_tr_combined, X_te, y_tr_combined, y_te = self._train_test_split(X, y)
            logger.info(f"Split data: train={len(X_tr_combined)}, test={len(X_te)}")
        
        # Encode labels
        y_tr_encoded, y_te_encoded, class_names = self._encode_labels(y_tr_combined, y_te)
        
        # Get predictions based on strategy
        if self.strategy == "linear_probe":
            y_pred, y_proba = self._linear_probe(X_tr_combined, y_tr_encoded, X_te)
        else:  # nearest_centroid
            y_pred, y_proba = self._nearest_centroid(X_tr_combined, y_tr_encoded, X_te)
        
        # Compute all metrics
        results = self._compute_metrics(
            y_true=y_te_encoded,
            y_pred=y_pred,
            y_proba=y_proba,
            class_names=class_names,
            n_train=len(y_tr_combined),
            n_test=len(y_te),
        )
        
        # Add metadata summary if provided
        if metadata:
            results["metadata_summary"] = self._summarize_metadata(metadata)
        
        # Add classifier if requested
        if return_classifier and hasattr(self, 'classifier'):
            results["classifier"] = self.classifier
            results["scaler"] = self.scaler
            results["label_encoder"] = self.label_encoder
        
        return results
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        exclude_unknown: bool = True,
        return_all_folds: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Embeddings array
            y: Labels array
            n_splits: Number of folds
            exclude_unknown: If True, exclude unknown labels
            return_all_folds: If True, return per-fold results
            
        Returns:
            Dictionary with cross-validation metrics
        """
        if exclude_unknown:
            known_mask = y != UNKNOWN_LABEL
            X = X[known_mask]
            y = y[known_mask]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        fold_metrics = []
        all_predictions = []
        all_true = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Encode labels
            y_tr_encoded, y_te_encoded, class_names = self._encode_labels(y_train, y_test)
            
            # Get predictions
            if self.strategy == "linear_probe":
                y_pred, y_proba = self._linear_probe(X_train, y_tr_encoded, X_test)
            else:
                y_pred, y_proba = self._nearest_centroid(X_train, y_tr_encoded, X_test)
            
            # Compute fold metrics
            fold_results = self._compute_metrics(
                y_true=y_te_encoded,
                y_pred=y_pred,
                y_proba=y_proba,
                class_names=class_names,
                n_train=len(y_train),
                n_test=len(y_test),
            )
            fold_results["fold"] = fold + 1
            fold_metrics.append(fold_results)
            
            all_predictions.extend(y_pred)
            all_true.extend(y_te_encoded)
        
        # Aggregate results
        aggregated = self._aggregate_cv_results(fold_metrics, all_true, all_predictions, class_names)
        
        if return_all_folds:
            aggregated["per_fold_results"] = fold_metrics
        
        return aggregated
    
    # ==================== Private Methods ====================
    
    def _train_test_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets, stratified by label."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        
        return X_train, X_test, y_train, y_test
    
    def _encode_labels(
        self, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Encode string labels to integers and store encoder.
        
        Returns:
            y_train_encoded, y_test_encoded, class_names
        """
        self.label_encoder = LabelEncoder()
        
        # Fit on training labels only
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        class_names = [str(c) for c in self.label_encoder.classes_]
        
        # Map numeric labels back to original values for reference
        label_map = {0: "normal", 1: "abnormal"}
        class_names_display = [label_map.get(int(c), f"class_{c}") for c in self.label_encoder.classes_]
        
        logger.info(f"Encoded classes: {class_names_display}")
        
        return y_train_encoded, y_test_encoded, class_names_display
    
    def _linear_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit logistic regression with standardization.
        
        Returns:
            y_pred, y_proba
        """
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = self.classifier.predict(X_test_scaled)
        y_proba = self.classifier.predict_proba(X_test_scaled)
        
        return y_pred, y_proba
    
    def _nearest_centroid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify by cosine distance to centroids.
        
        Returns:
            y_pred, y_scores (cosine similarities converted to [0,1])
        """
        from sklearn.preprocessing import normalize
        
        # L2 normalize for cosine distance
        X_train_norm = normalize(X_train, norm='l2')
        X_test_norm = normalize(X_test, norm='l2')
        
        # Train classifier
        self.classifier = NearestCentroid(metric='euclidean')
        self.classifier.fit(X_train_norm, y_train)
        
        # Predict
        y_pred = self.classifier.predict(X_test_norm)
        
        # Compute cosine similarities to centroids as scores
        centroids_norm = normalize(self.classifier.centroids_, norm='l2')
        y_scores = X_test_norm @ centroids_norm.T
        y_scores = (y_scores + 1) / 2  # Scale to [0, 1]
        
        return y_pred, y_scores
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        class_names: List[str],
        n_train: int,
        n_test: int,
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        
        # Basic metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Per-class F1
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1_dict = {
            name: float(score) 
            for name, score in zip(class_names, per_class_f1)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # AUROC (binary: normal vs abnormal)
        auroc = self._compute_binary_auroc(y_true, y_proba, class_names)
        
        # Detailed classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        results = {
            'strategy': self.strategy,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_f1': per_class_f1_dict,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': class_names,
            'auroc_binary': auroc,
            'n_train_samples': n_train,
            'n_test_samples': n_test,
            'classification_report': report,
        }
        
         # FIXED: Handle conditional formatting outside the f-string
        auroc_str = f"{auroc:.3f}" if auroc is not None else "N/A"
        logger.info(
        f"Results → accuracy={accuracy:.3f} | macro_f1={macro_f1:.3f} | "
        f"auroc={auroc_str}"
        )
        return results
    
    def _compute_binary_auroc(
        self,
        y_true: np.ndarray,
        y_proba: Optional[np.ndarray],
        class_names: List[str],
    ) -> Optional[float]:
        """
        Compute binary AUROC: normal (0) vs abnormal (1).
        
        Maps original labels (0=normal, 1=abnormal) to binary format.
        """
        if y_proba is None or len(np.unique(y_true)) < 2:
            return None
        
        try:
            # Map to binary: 0=normal, 1=abnormal
            y_binary = (y_true == 1).astype(int)
            
            if len(np.unique(y_binary)) < 2:
                logger.warning("Only one class present in test set for AUROC")
                return None
            
            # Get probability of abnormal class
            if len(class_names) == 2:
                # Find index of abnormal class (should be 1)
                abnormal_idx = 1 if class_names[1] == 'abnormal' else 0
                abnormal_scores = y_proba[:, abnormal_idx]
            else:
                # Multi-class case: use 1 - P(normal)
                normal_idx = 0 if class_names[0] == 'normal' else 1
                abnormal_scores = 1 - y_proba[:, normal_idx]
            
            return float(roc_auc_score(y_binary, abnormal_scores))
            
        except Exception as e:
            logger.warning(f"AUROC computation failed: {e}")
            return None
    
    def _aggregate_cv_results(
        self,
        fold_metrics: List[Dict],
        all_true: List,
        all_pred: List,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        
        # Mean and std of key metrics
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        aggregated = {
            f"{metric}_mean": float(np.mean([m[metric] for m in fold_metrics]))
            for metric in metrics
        }
        aggregated.update({
            f"{metric}_std": float(np.std([m[metric] for m in fold_metrics]))
            for metric in metrics
        })
        
        # Overall metrics on concatenated predictions
        aggregated['overall_accuracy'] = float(accuracy_score(all_true, all_pred))
        aggregated['overall_macro_f1'] = float(f1_score(all_true, all_pred, average='macro', zero_division=0))
        
        # Per-class F1 across all predictions
        per_class_f1 = f1_score(all_true, all_pred, average=None, zero_division=0)
        aggregated['per_class_f1'] = {
            name: float(score) 
            for name, score in zip(class_names, per_class_f1)
        }
        
        # Confusion matrix on all predictions
        aggregated['overall_confusion_matrix'] = confusion_matrix(all_true, all_pred).tolist()
        aggregated['confusion_matrix_labels'] = class_names
        
        # AUROC on all predictions
        # Note: This requires probability scores which we don't have aggregated
        aggregated['n_folds'] = len(fold_metrics)
        aggregated['n_total_samples'] = len(all_true)
        
        logger.info(
            f"CV Results (n_folds={len(fold_metrics)}): "
            f"acc={aggregated['accuracy_mean']:.3f}±{aggregated['accuracy_std']:.3f} | "
            f"macro_f1={aggregated['macro_f1_mean']:.3f}±{aggregated['macro_f1_std']:.3f}"
        )
        
        return aggregated
    
    def _summarize_metadata(self, metadata: List[Any]) -> Dict[str, Any]:
        """
        Summarize metadata statistics.
        Handles both dictionary and string metadata formats.
        """
        directories = {}
        files = {}
        
        for m in metadata:
            # Case 1: Metadata is a dictionary
            if isinstance(m, dict):
                # Count by directory
                dir_name = m.get('directory', 'unknown')
                directories[dir_name] = directories.get(dir_name, 0) + 1
                
                # Extract original file prefix
                file_name = m.get('file', 'unknown')
                prefix = file_name.split('_window')[0] if '_window' in file_name else file_name
                files[prefix] = files.get(prefix, 0) + 1
            
            # Case 2: Metadata is a string (filename)
            elif isinstance(m, str):
                # Try to parse directory from filename
                if '/' in m:
                    parts = m.split('/')
                    dir_name = parts[0] if len(parts) > 1 else 'unknown'
                else:
                    dir_name = 'unknown'
                
                directories[dir_name] = directories.get(dir_name, 0) + 1
                
                # Extract file prefix
                prefix = m.split('_window')[0] if '_window' in m else m
                files[prefix] = files.get(prefix, 0) + 1
            
            # Case 3: Other types - skip
            else:
                continue
        
        return {
            'n_unique_directories': len(directories),
            'directory_distribution': directories,
            'n_unique_files': len(files),
            'samples_per_file': {f: count for f, count in list(files.items())[:10]},  # First 10
            'metadata_type': str(type(metadata[0])) if metadata else 'unknown'
        }

# ==================== Helper Functions ====================

def _check_sklearn() -> None:
    """Verify scikit-learn is installed."""
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for PCGEmbeddingEvaluator. "
            "Run: pip install scikit-learn"
        ) from e


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example: Using with PCGEmbeddingDataset
    import sys
    sys.path.append('.')
    
    try:
        from data_loading import PCGEmbeddingDataset
        
        print("="*60)
        print("PCG Embedding Evaluator - Example Usage")
        print("="*60)
        
        # 1. Load data using your dataloader
        print("\n1. Loading embeddings...")
        dataset = PCGEmbeddingDataset(include_unknown=False)
        
        # Get all data
        X = dataset.embeddings  # (n_samples, 256)
        y = dataset.labels      # (n_samples,) with values 0, 1
        metadata = dataset.metadata
        
        print(f"   X shape: {X.shape}")
        print(f"   y distribution: normal={sum(y==0)}, abnormal={sum(y==1)}")
        
        # 2. Create evaluator
        print("\n2. Creating evaluator...")
        evaluator = PCGEmbeddingEvaluator(
            strategy='linear_probe',
            random_state=42,
            test_size=0.2
        )
        
        # 3. Run evaluation
        print("\n3. Running evaluation...")
        results = evaluator.evaluate(
            X=X,
            y=y,
            metadata=metadata
        )
        
        # 4. Print results
        print("\n4. Results:")
        print(f"   Accuracy: {results['accuracy']:.3f}")
        print(f"   Macro F1: {results['macro_f1']:.3f}")
        print(f"   Binary AUROC: {results['auroc_binary']:.3f}")
        print(f"\n   Per-class F1:")
        for class_name, f1 in results['per_class_f1'].items():
            print(f"     {class_name}: {f1:.3f}")
        
        # 5. Run cross-validation
        print("\n5. Running 5-fold cross-validation...")
        cv_results = evaluator.cross_validate(
            X=X,
            y=y,
            n_splits=5
        )
        
        print(f"\n   CV Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
        print(f"   CV Macro F1: {cv_results['macro_f1_mean']:.3f} ± {cv_results['macro_f1_std']:.3f}")
        
    except ImportError:
        print("\nNote: To run the example, ensure pcg_embedding_dataloader.py is in your path")
        print("or run this after creating the dataloader script.")