import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing
import tqdm

class LatentDistribution:
    """Class to handle latent distributions with mean and logvar."""
    def __init__(self, mean, logvar):
        self.mean = mean
        self.logvar = logvar
        
    def sample(self):
        """Sample from the latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mean + eps * std

class VideoEmbeddingDataset(Dataset):
    """Dataset for loading video latents and caption embeddings."""
    
    def __init__(
        self, 
        data_dir: str,
        caption_dir: Optional[str] = None,
        file_extension: str = ".latent.pt",
        caption_extension: str = ".embed.pt",
        device: str = "cpu",
        use_bfloat16: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing video latent files
            caption_dir: Directory containing caption embedding files. If None, will be derived from data_dir
            file_extension: Extension of latent files
            caption_extension: Extension of caption embedding files
            device: Device to load tensors to
            use_bfloat16: Whether to convert tensors to bfloat16
        """
        self.data_dir = data_dir
        self.caption_dir = caption_dir or os.path.join(os.path.dirname(data_dir), "captions")
        self.file_extension = file_extension
        self.caption_extension = caption_extension
        self.device = device
        self.use_bfloat16 = use_bfloat16
        
        # Get all latent files
        self.file_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(file_extension):
                    self.file_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.file_paths)} video latent files in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load video latent
        file_path = self.file_paths[idx]
        latent_dict = torch.load(file_path, map_location="cpu")
        
        # Create latent distribution from mean and logvar
        ldist = LatentDistribution(latent_dict["mean"], latent_dict["logvar"])
        # Sample from the distribution
        z_0 = ldist.sample()
        
        # Derive and load corresponding caption embedding
        rel_path = os.path.relpath(file_path, self.data_dir)
        caption_path = os.path.join(self.caption_dir, rel_path).replace(self.file_extension, self.caption_extension)
        caption_dict = torch.load(caption_path, map_location="cpu")
        # print("caption_path", caption_path,"\nfile_path", file_path)
        
        # Extract caption features and mask (assuming batch size 1 in the saved embeddings)
        y_feat = caption_dict["y_feat"][0]
        y_mask = caption_dict["y_mask"][0]

        
        return {
            "z_0": z_0,
            "y_feat": y_feat,
            "y_mask": y_mask,
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle batching of samples.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Dictionary with batched tensors
        """
        z_0 = torch.cat([item["z_0"] for item in batch], dim=0)
        y_feat = torch.cat([item["y_feat"] for item in batch], dim=0)
        y_mask = torch.cat([item["y_mask"] for item in batch], dim=0)
        
        # We'll handle device placement and dtype conversion in the main process
        # after pin_memory if needed, not here in the collate function
        
        return {
            "z_0": z_0,
            "y_feat": y_feat,
            "y_mask": y_mask,
        }

def get_video_embedding_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    use_bfloat16: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for video embeddings.
    
    Args:
        data_dir: Directory containing video latent files
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader
        device: Device to load tensors to
        use_bfloat16: Whether to convert tensors to bfloat16
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader for video embeddings
    """
    dataset = VideoEmbeddingDataset(
        data_dir=data_dir,
        device="cpu",  # Always load to CPU first
        use_bfloat16=False,  # Don't convert to bfloat16 in the dataset
    )
    
    # When using CUDA with multiprocessing, we need to be careful about device placement
    use_cuda = device.startswith("cuda")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if not use_cuda else 0,  # Use 0 workers with CUDA for testing
        collate_fn=dataset.collate_fn,
        pin_memory=use_cuda,  # Use pin_memory when using CUDA
    )

if __name__ == "__main__":
    # Example usage and testing
    import argparse
    import multiprocessing
    import tqdm
    
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Test VideoEmbeddingDataset")
    parser.add_argument("--data_dir", type=str, 
                        default="",
                        help="Directory containing video latent files")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to load tensors to")
    parser.add_argument("--test_all", action="store_true", help="Test all dataset items for integrity")
    args = parser.parse_args()
    
    print(f"Testing VideoEmbeddingDataset with data from {args.data_dir}")
    
    # Create dataset and dataloader
    dataloader = get_video_embedding_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        use_bfloat16=True,
    )
    
    # Get a batch of data
    print(f"Fetching a batch from dataloader...")
    batch = next(iter(dataloader))
    
    # Move to device and convert to bfloat16 if needed
    device = torch.device(args.device)
    use_bfloat16 = True
    
    if use_bfloat16 and device.type == "cuda":
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            batch["z_0"] = batch["z_0"].to(device)
            batch["y_feat"] = batch["y_feat"].to(device)
            batch["y_mask"] = batch["y_mask"].to(device)
    else:
        batch["z_0"] = batch["z_0"].to(device)
        batch["y_feat"] = batch["y_feat"].to(device)
        batch["y_mask"] = batch["y_mask"].to(device)
    
    # Add conditioning dictionary
    batch["conditioning"] = {
        "cond": {
            "y_feat": [batch["y_feat"]],
            "y_mask": [batch["y_mask"]]
        }
    }
    
    # Print batch information
    print(f"Batch keys: {batch.keys()}")
    print(f"z_0 shape: {batch['z_0'].shape}, dtype: {batch['z_0'].dtype}")
    print(f"y_feat shape: {batch['y_feat'].shape}, dtype: {batch['y_feat'].dtype}")
    print(f"y_mask shape: {batch['y_mask'].shape}, dtype: {batch['y_mask'].dtype}")
    print(f"conditioning keys: {batch['conditioning'].keys()}")
    print(f"conditioning['cond'] keys: {batch['conditioning']['cond'].keys()}")
    
    # Test all dataset items if requested
    if args.test_all:
        # Create dataset
        dataset = VideoEmbeddingDataset(
            data_dir=args.data_dir,
            device="cpu",  # Use CPU for initial file checking
        )
        
        if len(dataset) == 0:
            print("Dataset is empty!")
            exit(0)
            
        print(f"\nTesting all {len(dataset)} dataset items for integrity...")
        broken_items = []
        missing_captions = []
        
        # First check for missing caption files (faster than loading batches)
        print("Checking for missing caption files...")
        for idx in tqdm.tqdm(range(len(dataset))):
            file_path = dataset.file_paths[idx]
            caption_path = file_path.replace("videos_prepared", "captions").replace(
                dataset.file_extension, dataset.caption_extension)
            
            # Check if caption file exists
            if not os.path.exists(caption_path):
                missing_captions.append((idx, file_path, caption_path))
        
        # Now test loading in batches
        print("Testing data loading in batches...")
        # Create a dataloader with the specified batch size
        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        
        for batch_idx, batch_indices in enumerate(range(0, len(dataset), args.batch_size)):
            batch_end = min(batch_indices + args.batch_size, len(dataset))
            indices = list(range(batch_indices, batch_end))
            
            try:
                # Try to load the batch
                batch = next(iter(torch.utils.data.DataLoader(
                    torch.utils.data.Subset(dataset, indices),
                    batch_size=len(indices),
                    shuffle=False,
                    num_workers=0  # Use single process for error tracking
                )))
                
                # Check for NaN values in the batch
                if torch.isnan(batch["z_0"]).any() or torch.isnan(batch["y_feat"]).any():
                    # If NaNs found, check individual samples to identify which ones are problematic
                    for i, idx in enumerate(indices):
                        if (torch.isnan(batch["z_0"][i]).any() or 
                            torch.isnan(batch["y_feat"][i]).any()):
                            broken_items.append((idx, dataset.file_paths[idx], "Contains NaN values"))
                
            except Exception as e:
                # If batch loading fails, try individual items to identify which ones are problematic
                for idx in indices:
                    try:
                        file_path = dataset.file_paths[idx]
                        item = dataset[idx]
                        # Verify tensor shapes and types
                        if not all(k in item for k in ["z_0", "y_feat", "y_mask"]):
                            broken_items.append((idx, file_path, "Missing keys"))
                        elif torch.isnan(item["z_0"]).any() or torch.isnan(item["y_feat"]).any():
                            broken_items.append((idx, file_path, "Contains NaN values"))
                    except Exception as item_e:
                        broken_items.append((idx, dataset.file_paths[idx], str(item_e)))
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_end}/{len(dataset)} items. "
                      f"Found {len(broken_items)} broken items, {len(missing_captions)} missing captions.")
        
        # Report results
        print(f"\nIntegrity test completed.")
        print(f"Found {len(broken_items)} broken items.")
        print(f"Found {len(missing_captions)} items with missing caption files.")
        
        if broken_items:
            print("\nBroken items:")
            for idx, path, reason in broken_items[:20]:  # Show first 20
                print(f"  {idx}: {path} - {reason}")
            if len(broken_items) > 20:
                print(f"  ... and {len(broken_items) - 20} more")
                
        if missing_captions:
            print("\nMissing caption files:")
            for idx, video_path, caption_path in missing_captions[:20]:  # Show first 20
                print(f"  {idx}: Missing {caption_path}")
            if len(missing_captions) > 20:
                print(f"  ... and {len(missing_captions) - 20} more")
    
    print("\nTest completed successfully!")
