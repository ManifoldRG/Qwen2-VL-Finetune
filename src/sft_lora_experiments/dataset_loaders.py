#!/usr/bin/env python3
"""
Dataset loading utilities for Hugging Face datasets.

Provides functions to load datasets in streaming or non-streaming mode
with consistent error handling and logging.
"""

from datasets import load_dataset
from typing import Any, Union
from datasets.iterable_dataset import IterableDataset
from datasets.arrow_dataset import Dataset


def load_grounding_dataset(streaming: bool = True) -> Union[IterableDataset, Dataset]:
    """
    Load Salesforce grounding dataset from Hugging Face.
    
    Args:
        streaming: If True, use streaming mode (default). If False, download entire dataset (~34.7 GB).
    
    Returns:
        Dataset object (IterableDataset if streaming, Dataset otherwise)
    
    Raises:
        Exception: If dataset loading fails
    """
    print("Loading Salesforce/grounding_dataset from Hugging Face...")
    if streaming:
        print("Using streaming mode (no full download required)")
        print("NOTE: Images are downloaded on-demand when accessed and cached in ~/.cache/huggingface/")
        print("      This is expected behavior - cache will grow as you process samples.")
        try:
            dataset = load_dataset("Salesforce/grounding_dataset", streaming=True, split="train")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    else:
        print("WARNING: Downloading entire dataset (~34.7 GB). This may take a while...")
        dataset = load_dataset("Salesforce/grounding_dataset", split="train")
        return dataset


def load_autogui_dataset(streaming: bool = True) -> Union[IterableDataset, Dataset]:
    """
    Load AutoGUI dataset from Hugging Face.
    
    Args:
        streaming: If True, use streaming mode (default). If False, download entire dataset.
    
    Returns:
        Dataset object (IterableDataset if streaming, Dataset otherwise)
    
    Raises:
        Exception: If dataset loading fails
    """
    print("Loading AutoGUI dataset from Hugging Face...")
    if streaming:
        print("Using streaming mode (no full download required)")
    else:
        print("WARNING: Downloading entire dataset. This may take a while...")
    
    try:
        if streaming:
            dataset = load_dataset("AutoGUI/AutoGUI-v1-702k", streaming=True, split="train")
        else:
            dataset = load_dataset("AutoGUI/AutoGUI-v1-702k", split="train")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

