#!/usr/bin/env python3
"""
Sampling strategies for dataset conversion.

Provides abstract base class and concrete implementations for different
sampling strategies (stratified, reservoir, collect-all) to eliminate
code duplication.
"""

import random
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.
    
    Handles common logic: progress logging, filter tracking, sample validation.
    Subclasses implement specific sampling algorithms.
    """
    
    def __init__(
        self,
        extract_metadata_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        is_valid_fn: Callable[[Dict[str, Any]], bool],
        get_filter_reason_fn: Callable[[Dict[str, Any], Optional[str]], Optional[str]],
        rng: random.Random,
        dataset_filter: Optional[str] = None
    ):
        """
        Initialize sampling strategy.
        
        Args:
            extract_metadata_fn: Function to extract metadata from a sample
            is_valid_fn: Function to check if a sample is valid
            get_filter_reason_fn: Function to get filter reason for a sample
            rng: Random number generator
            dataset_filter: Optional dataset filter string
        """
        self.extract_metadata_fn = extract_metadata_fn
        self.is_valid_fn = is_valid_fn
        self.get_filter_reason_fn = get_filter_reason_fn
        self.rng = rng
        self.dataset_filter = dataset_filter
        
        # Common state
        self.samples_processed = 0
        self.samples_filtered = 0
        self.filter_reasons = Counter()
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def _should_log_progress(self, current_time: float) -> bool:
        """Check if progress should be logged."""
        return (
            self.samples_processed % 1000 == 0 or
            (current_time - self.last_log_time) >= 10
        )
    
    def _log_progress(self, current_time: float, valid_count: int) -> None:
        """Log progress information."""
        elapsed = current_time - self.start_time
        rate = self.samples_processed / elapsed if elapsed > 0 else 0
        print(
            f"Processed {self.samples_processed:,} samples "
            f"(filtered: {self.samples_filtered:,}, valid: {valid_count}) | "
            f"Rate: {rate:.1f} samples/sec | Elapsed: {elapsed:.1f}s"
        )
        if self.filter_reasons:
            print(f"  Filter reasons: {dict(self.filter_reasons.most_common(5))}")
        self.last_log_time = current_time
    
    def _process_sample(
        self,
        sample: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Process a single sample: validate, filter, extract metadata.
        
        Returns:
            (sample_data_dict, filter_reason)
            - sample_data_dict: Dict with 'sample' and 'metadata' if valid, None otherwise
            - filter_reason: Reason for filtering if filtered, None otherwise
        """
        # Check filter reason
        filter_reason = self.get_filter_reason_fn(sample, self.dataset_filter)
        if filter_reason:
            self.filter_reasons[filter_reason] += 1
            return None, filter_reason
        
        # Validate sample
        if not self.is_valid_fn(sample):
            self.filter_reasons["unknown"] += 1
            return None, "unknown"
        
        # Extract metadata
        try:
            metadata = self.extract_metadata_fn(sample)
            return {
                "sample": sample,
                "metadata": metadata
            }, None
        except Exception as e:
            # Extract sample identifier for better error messages
            sample_id = sample.get("uuid") or sample.get("id") or f"sample_{self.samples_processed}"
            error_reason = f"extract_metadata_error ({type(e).__name__})"
            self.filter_reasons[error_reason] += 1
            print(f"Warning: Failed to extract metadata for {sample_id}: {e}")
            return None, error_reason
    
    def _iterate_dataset(self, dataset) -> None:
        """
        Common iteration logic over dataset.
        
        Calls _process_sample for each sample and delegates to
        _handle_valid_sample for valid samples.
        """
        try:
            for sample in dataset:
                self.samples_processed += 1
                current_time = time.time()
                
                # Progress logging
                if self._should_log_progress(current_time):
                    self._log_progress(current_time, self._get_valid_count())
                
                # Process sample
                sample_data, filter_reason = self._process_sample(sample)
                if sample_data is None:
                    self.samples_filtered += 1
                    continue
                
                # Handle valid sample (strategy-specific)
                self._handle_valid_sample(sample_data)
                
                # Check for early termination (strategy-specific)
                if self._should_terminate():
                    break
        except StopIteration:
            # Dataset iterator exhausted (non-streaming dataset finished)
            pass
        except Exception as e:
            raise
    
    @abstractmethod
    def _get_valid_count(self) -> int:
        """Get current count of valid samples collected."""
        pass
    
    @abstractmethod
    def _handle_valid_sample(self, sample_data: Dict[str, Any]) -> None:
        """Handle a valid sample (strategy-specific)."""
        pass
    
    @abstractmethod
    def _should_terminate(self) -> bool:
        """Check if sampling should terminate early."""
        pass
    
    @abstractmethod
    def collect_samples(self, dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
        """
        Collect samples using this strategy.
        
        Args:
            dataset: Dataset to sample from
            args: Arguments object with sampling configuration
        
        Returns:
            List of sample data dictionaries
        """
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "samples_processed": self.samples_processed,
            "samples_filtered": self.samples_filtered,
            "filter_reasons": dict(self.filter_reasons),
            "valid_samples": self._get_valid_count()
        }


class StratifiedSamplingStrategy(SamplingStrategy):
    """Stratified sampling by dataset source."""
    
    def __init__(self, *args, max_first_pass_samples: int = 20000, min_samples_per_dataset: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_by_dataset = defaultdict(list)
        self.dataset_counts = Counter()
        self.max_first_pass_samples = max_first_pass_samples
        self.min_samples_per_dataset = min_samples_per_dataset
    
    def _get_valid_count(self) -> int:
        return sum(len(samples) for samples in self.samples_by_dataset.values())
    
    def _handle_valid_sample(self, sample_data: Dict[str, Any]) -> None:
        metadata = sample_data["metadata"]
        dataset_name = metadata.get("dataset", "unknown")
        self.samples_by_dataset[dataset_name].append(sample_data)
        self.dataset_counts[dataset_name] += 1
    
    def _should_terminate(self) -> bool:
        # Stop if we've processed too many samples (streaming dataset protection)
        if self.samples_processed >= self.max_first_pass_samples:
            return True
        # Stop if we have enough samples from each dataset (early termination optimization)
        # For stratified sampling, we need enough samples to calculate proportions
        # Stop early if we have at least 10k samples and each dataset has at least min_samples_per_dataset
        if len(self.samples_by_dataset) > 0 and self.samples_processed >= 10000:
            min_samples = min(len(samples) for samples in self.samples_by_dataset.values())
            if min_samples >= self.min_samples_per_dataset:
                return True
        return False
    
    def collect_samples(self, dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
        """Collect samples using stratified sampling."""
        print("First pass: collecting samples by dataset...")
        print(f"  (Will stop after {self.max_first_pass_samples} samples or when each dataset has {self.min_samples_per_dataset} samples)")
        self._iterate_dataset(dataset)
        
        print(f"\nCollected samples by dataset:")
        for dataset_name, count in self.dataset_counts.most_common():
            print(f"  {dataset_name}: {count}")
        
        # Stratified sampling: sample proportionally from each dataset
        if args.num_samples:
            total_available = sum(len(samples) for samples in self.samples_by_dataset.values())
            if args.num_samples > total_available:
                print(
                    f"Warning: Requested {args.num_samples} samples but only {total_available} available. "
                    f"Using all samples."
                )
                args.num_samples = total_available
            
            # Calculate quotas per dataset (proportional to their representation)
            quotas = {}
            for dataset_name, samples in self.samples_by_dataset.items():
                proportion = len(samples) / total_available
                quotas[dataset_name] = max(1, int(args.num_samples * proportion))
            
            # Adjust quotas to sum to exactly num_samples
            total_quota = sum(quotas.values())
            if total_quota != args.num_samples:
                diff = args.num_samples - total_quota
                # Add difference to largest dataset (deterministic tie-breaking by name)
                largest_dataset = max(quotas.items(), key=lambda x: (x[1], x[0]))[0]
                quotas[largest_dataset] += diff
            
            print(f"\nStratified sampling quotas:")
            for dataset_name, quota in sorted(quotas.items()):
                available = len(self.samples_by_dataset[dataset_name])
                print(f"  {dataset_name}: {quota} (from {available} available)")
            
            # Sample from each dataset
            valid_samples = []
            for dataset_name, samples in self.samples_by_dataset.items():
                quota = quotas.get(dataset_name, 0)
                if quota > 0:
                    if len(samples) <= quota:
                        valid_samples.extend(samples)
                    else:
                        sampled = self.rng.sample(samples, quota)
                        valid_samples.extend(sampled)
            return valid_samples
        else:
            # Use all samples
            valid_samples = []
            for samples in self.samples_by_dataset.values():
                valid_samples.extend(samples)
            return valid_samples


class ReservoirSamplingStrategy(SamplingStrategy):
    """Reservoir sampling for fixed-size random sample."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reservoir: List[Dict[str, Any]] = []
        self.items_seen = 0
        self.num_samples: Optional[int] = None
        self.min_samples_to_process: Optional[int] = None
    
    def _get_valid_count(self) -> int:
        return len(self.reservoir)
    
    def _handle_valid_sample(self, sample_data: Dict[str, Any]) -> None:
        if self.num_samples is None:
            return  # Not initialized yet
        
        self.items_seen += 1
        
        # Reservoir sampling algorithm
        if len(self.reservoir) < self.num_samples:
            # Reservoir not full: add sample
            self.reservoir.append(sample_data)
        else:
            # Reservoir full: replace with probability k/i
            j = self.rng.randint(0, self.items_seen - 1)
            if j < self.num_samples:
                self.reservoir[j] = sample_data
    
    def _should_terminate(self) -> bool:
        if self.num_samples is None or self.min_samples_to_process is None:
            return False
        return (
            len(self.reservoir) >= self.num_samples and
            self.samples_processed >= self.min_samples_to_process
        )
    
    def collect_samples(self, dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
        """Collect samples using reservoir sampling."""
        self.num_samples = args.num_samples
        self.min_samples_to_process = args.num_samples * 10
        
        print(f"Using reservoir sampling to collect {self.num_samples} samples...")
        print(
            f"This will stop early once enough valid samples are found "
            f"(max ~{self.min_samples_to_process} samples processed)."
        )
        
        self._iterate_dataset(dataset)
        
        if len(self.reservoir) >= self.num_samples:
            print(f"Collected {len(self.reservoir)} valid samples. Stopping early.")
        
        return self.reservoir


class CollectAllSamplingStrategy(SamplingStrategy):
    """Collect all valid samples."""
    
    def __init__(self, *args, max_samples: Optional[int] = 100000, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_valid: List[Dict[str, Any]] = []
        self.max_samples = max_samples
    
    def _get_valid_count(self) -> int:
        return len(self.all_valid)
    
    def _handle_valid_sample(self, sample_data: Dict[str, Any]) -> None:
        self.all_valid.append(sample_data)
    
    def _should_terminate(self) -> bool:
        # For streaming datasets, we need a safety limit
        # This prevents infinite iteration on streaming datasets
        # max_samples=0 means no limit (only safe with non-streaming datasets)
        if self.max_samples is not None and self.max_samples > 0:
            return self.samples_processed >= self.max_samples
        return False
    
    def collect_samples(self, dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
        """Collect all valid samples."""
        # max_samples is set in __init__ via kwargs
        if self.max_samples and self.max_samples > 0:
            print(f"Collecting all samples (max {self.max_samples:,} for streaming safety)...")
        else:
            print("Collecting all samples (no limit - requires non-streaming dataset)...")
        self._iterate_dataset(dataset)
        return self.all_valid


def create_sampling_strategy(
    args: argparse.Namespace,
    extract_metadata_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    is_valid_fn: Callable[[Dict[str, Any]], bool],
    get_filter_reason_fn: Callable[[Dict[str, Any], Optional[str]], Optional[str]],
    rng: random.Random
) -> SamplingStrategy:
    """
    Factory function to create appropriate sampling strategy based on args.
    
    Args:
        args: Arguments object with sampling configuration (must have: stratified, num_samples, dataset_filter, max_first_pass_samples, min_samples_per_dataset, max_collect_all_samples)
        extract_metadata_fn: Function to extract metadata from a sample
        is_valid_fn: Function to check if a sample is valid
        get_filter_reason_fn: Function to get filter reason for a sample
        rng: Random number generator
    
    Returns:
        Appropriate SamplingStrategy instance
    """
    common_kwargs = {
        "extract_metadata_fn": extract_metadata_fn,
        "is_valid_fn": is_valid_fn,
        "get_filter_reason_fn": get_filter_reason_fn,
        "rng": rng,
        "dataset_filter": args.dataset_filter
    }
    
    if args.stratified:
        return StratifiedSamplingStrategy(
            **common_kwargs,
            max_first_pass_samples=getattr(args, 'max_first_pass_samples', 20000),
            min_samples_per_dataset=getattr(args, 'min_samples_per_dataset', 50)
        )
    elif args.num_samples:
        return ReservoirSamplingStrategy(**common_kwargs)
    else:
        return CollectAllSamplingStrategy(
            **common_kwargs,
            max_samples=getattr(args, 'max_collect_all_samples', 100000)
        )

