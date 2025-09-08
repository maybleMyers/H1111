# FlowMatch Pusa Scheduler
# Based on the Pusa extension for enhanced video generation quality
# Copyright 2024-2025 Implementation for Wan2_2 Pipeline

import torch
import numpy as np
from typing import Optional, Union


class FlowMatchSchedulerPusa:
    """
    FlowMatch scheduler specifically designed for Pusa extensions.
    Implements custom sigma scheduling with noise multiplier support.
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
        sigmas: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Set timesteps for the scheduler.
        
        Args:
            num_inference_steps: Number of inference steps
            denoising_strength: Strength of denoising (0.0 to 1.0)
            training: Whether in training mode (enables special weighting)
            shift: Override shift parameter
            sigmas: Custom sigma schedule
            device: Target device for tensors
        """
        if shift is not None:
            self.shift = shift
            
        if device is None:
            device = torch.device("cpu")
            
        # Calculate sigma start based on denoising strength
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        
        if sigmas is None:
            steps = num_inference_steps
            if self.extra_one_step:
                self.sigmas = torch.linspace(sigma_start, self.sigma_min, steps, device=device)[:-1]
            else:
                self.sigmas = torch.linspace(sigma_start, self.sigma_min, steps, device=device)
                
            if self.inverse_timesteps:
                self.sigmas = torch.flip(self.sigmas, dims=[0])
        else:
            if isinstance(sigmas, np.ndarray):
                sigmas = torch.from_numpy(sigmas)
            self.sigmas = sigmas.to(device=device, dtype=torch.float32)
            
        # Apply shift transformation
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
            
        # Convert sigmas to timesteps
        self.timesteps = self.sigmas * self.num_train_timesteps
        
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
        noise_multipliers: Optional[float] = None,
        **kwargs
    ):
        """
        Perform one step of the Pusa scheduler.
        
        Args:
            model_output: The direct output from the model
            timestep: Current timestep
            sample: Current sample (latent)
            return_dict: Whether to return a dict or tuple
            generator: Random number generator (for compatibility)
            noise_multipliers: Pusa noise multipliers
            **kwargs: Additional arguments
            
        Returns:
            Previous sample (latent) for the next step
        """
        # Ensure tensors are on the same device
        if isinstance(timestep, torch.Tensor):
            self.timesteps = self.timesteps.to(timestep.device)
            self.sigmas = self.sigmas.to(timestep.device)
            model_output = model_output.to(timestep.device)
            sample = sample.to(timestep.device)
            
        # Handle batch timesteps
        if isinstance(timestep, torch.Tensor) and len(timestep.shape) == 1:
            timestep = timestep[0]
            
        # Find current step index
        step_index = None
        for i, t in enumerate(self.timesteps):
            if torch.allclose(t, timestep, atol=1e-4):
                step_index = i
                break
                
        if step_index is None:
            # Fallback: find closest timestep
            step_index = torch.argmin(torch.abs(self.timesteps - timestep)).item()
            
        # Get current and next sigma
        sigma = self.sigmas[step_index]
        if step_index < len(self.sigmas) - 1:
            sigma_next = self.sigmas[step_index + 1]
        else:
            sigma_next = torch.tensor(0.0, device=sample.device)
            
        # Euler step for flow matching
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        
        # Apply Pusa noise multipliers if provided
        if noise_multipliers is not None and noise_multipliers > 0:
            # Add scaled noise based on current sigma
            noise = torch.randn_like(prev_sample, generator=generator)
            noise_scale = noise_multipliers * sigma * 0.01  # Scale factor
            prev_sample = prev_sample + noise_scale * noise
            
        if not return_dict:
            return (prev_sample,)
        
        # Return compatible format
        from types import SimpleNamespace
        return SimpleNamespace(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to the noise schedule.
        
        Args:
            original_samples: Original clean samples
            noise: Noise to be added
            timesteps: Timesteps for noise addition
            
        Returns:
            Noisy samples
        """
        # Convert timesteps to sigmas
        sigmas = timesteps.float() / self.num_train_timesteps
        sigmas = sigmas.to(device=original_samples.device)
        
        # Add noise according to flow matching formulation
        noisy_samples = (1 - sigmas.view(-1, 1, 1, 1, 1)) * original_samples + sigmas.view(-1, 1, 1, 1, 1) * noise
        
        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get velocity for flow matching training.
        
        Args:
            sample: Clean samples
            noise: Noise samples  
            timesteps: Timesteps
            
        Returns:
            Velocity targets
        """
        return noise - sample

    def __len__(self):
        return len(self.timesteps) if self.timesteps is not None else 0