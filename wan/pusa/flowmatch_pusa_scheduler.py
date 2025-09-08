# FlowMatch Pusa Scheduler
# Based on the original Pusa extension implementation
# Copyright 2024-2025 Implementation for wan2_generate_video.py

import torch
import numpy as np
from typing import Optional, Union


class FlowMatchSchedulerPusa:
    """
    FlowMatch scheduler specifically designed for Pusa extensions.
    This implementation matches the original Pusa scheduler from the source repository.
    """
    
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ):
        """
        Initialize FlowMatchSchedulerPusa.
        
        Args:
            num_inference_steps: Number of inference steps for sampling
            num_train_timesteps: Number of training timesteps (typically 1000)
            shift: Shift parameter for sigma transformation
            sigma_max: Maximum sigma value
            sigma_min: Minimum sigma value
            inverse_timesteps: Whether to inverse the timestep order
            extra_one_step: Whether to exclude the last step in linspace
            reverse_sigmas: Whether to reverse sigma values (1 - sigma)
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        
        # Will be set by set_timesteps
        self.timesteps = None
        self.sigmas = None
        self.linear_timesteps_weights = None

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Set timesteps for the scheduler.
        
        Args:
            num_inference_steps: Number of inference steps
            denoising_strength: Strength of denoising (0.0 to 1.0)
            training: Whether in training mode (enables special weighting)
            shift: Override shift parameter
            device: Target device for tensors
        """
        if shift is not None:
            self.shift = shift
            
        # Calculate sigma start based on denoising strength
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
            
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
            
        # Apply shift transformation - this is the key Pusa transformation
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
            
        # Convert sigmas to timesteps
        self.timesteps = self.sigmas * self.num_train_timesteps
        
        # Move to device if specified
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
        
        # Training mode weighting (experimental)
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        to_final: bool = False,
        **kwargs
    ):
        """
        Perform one step of the Pusa scheduler.
        This implementation matches the original Pusa scheduler step method.
        
        Args:
            model_output: The direct output from the model
            timestep: Current timestep
            sample: Current sample (latent)
            return_dict: Whether to return a dict or tuple
            generator: Random number generator (for compatibility)
            to_final: Whether this is the final step
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Previous sample (latent) for the next step
        """
        # Ensure tensors are on the same device
        if isinstance(timestep, torch.Tensor):
            self.timesteps = self.timesteps.to(timestep.device)
            self.sigmas = self.sigmas.to(timestep.device)
            model_output = model_output.to(timestep.device)
            sample = sample.to(timestep.device)
            
        if len(timestep.shape) == 1:
            # Single timestep case
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
            sigma = self.sigmas[timestep_id]
            if to_final or timestep_id + 1 >= len(self.timesteps):
                sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
            else:
                sigma_ = self.sigmas[timestep_id + 1]
            prev_sample = sample + model_output * (sigma_ - sigma)
        else:
            # Batch timestep case
            timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep).abs(), dim=0)
            sigma = self.sigmas[timestep_id].unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).to(sample.device)
            
            # Handle sigma_ calculation for each timestep_id element
            if to_final or torch.any(timestep_id + 1 >= len(self.timesteps)):
                default_value = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
                # Create sigma_ with the same dtype as self.sigmas
                sigma_ = torch.ones_like(timestep_id, dtype=self.sigmas.dtype, device=sample.device) * default_value
                valid_indices = timestep_id + 1 < len(self.timesteps)
                if torch.any(valid_indices):
                    # Convert indices to the appropriate type for indexing
                    valid_timestep_ids = timestep_id[valid_indices] 
                    sigma_[valid_indices] = self.sigmas[(valid_timestep_ids + 1).to(torch.long)]
            else:
                sigma_ = self.sigmas[(timestep_id + 1).to(torch.long)]
                
            # Reshape sigma_ to match sigma's dimensions for the operation
            sigma_ = sigma_.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).to(sample.device)
            if torch.any(timestep == 0):
                zero_indices = torch.where(timestep == 0)[1].to(torch.long)
                sigma[:,:,zero_indices] = 0
                
            prev_sample = sample + model_output * (sigma_ - sigma)
            
        if not return_dict:
            return (prev_sample,)
        
        # Return compatible format
        from types import SimpleNamespace
        return SimpleNamespace(prev_sample=prev_sample)

    def return_to_timestep(self, timestep, sample, sample_stablized):
        """Return to a specific timestep (for advanced sampling)"""
        if isinstance(timestep, torch.Tensor):
            self.timesteps = self.timesteps.to(timestep.device)
            self.sigmas = self.sigmas.to(timestep.device)
        if len(timestep.shape) == 1:
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
            sigma = self.sigmas[timestep_id]
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep).abs(), dim=0)
            sigma = self.sigmas[timestep_id].unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).to(sample.device)
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to the noise schedule.
        
        Args:
            original_samples: Original clean samples
            noise: Noise to be added
            timestep: Timestep for noise addition
            
        Returns:
            Noisy samples
        """
        if isinstance(timestep, torch.Tensor):
            self.timesteps = self.timesteps.to(timestep.device)
            self.sigmas = self.sigmas.to(timestep.device)
        if len(timestep.shape) == 1:
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
            sigma = self.sigmas[timestep_id]
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(-1).unsqueeze(-1) - timestep.unsqueeze(0)).abs(), dim=0)
            sigma = self.sigmas[timestep_id].unsqueeze(1).unsqueeze(3).unsqueeze(4).to(original_samples.device)
        sample = (1 - sigma) * original_samples + sigma * noise
        
        return sample

    def training_target(self, sample, noise, timestep):
        """Get training target for flow matching"""
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """Get training weight for this timestep"""
        if isinstance(timestep, torch.Tensor):
            self.timesteps = self.timesteps.to(timestep.device)
            self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        if len(timestep.shape) == 1:
            timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.to(self.timesteps.device)).abs(), dim=0) 
        weights = self.linear_timesteps_weights[timestep_id].to(self.timesteps.device)
        return weights

    def __len__(self):
        return len(self.timesteps) if self.timesteps is not None else 0