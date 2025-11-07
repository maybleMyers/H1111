"""
Step Distillation Scheduler for lightx2v LoRA support
Based on WanStepDistillScheduler from lightx2v
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple


class StepDistillScheduler:
    """
    Step Distillation Scheduler that works with lightx2v distilled models.
    This scheduler uses a specific timestep schedule optimized for distilled models.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        **kwargs
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting

        # Flow matching parameters
        self.sigma_min = 0.0
        self.sigma_max = 1.0

        # Will be set by set_timesteps
        self.timesteps = None
        self.sigmas = None
        self.denoising_step_list = None
        self.denoising_step_index = None
        self.num_inference_steps = None
        self._step_index = None
        self._begin_index = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cuda",
        shift: Optional[float] = None
    ):
        """
        Set the timesteps for inference based on the number of steps.

        Args:
            num_inference_steps: Number of denoising steps (e.g., 4 for 4-step distillation)
            device: Device to place timesteps on
            shift: Optional shift factor override
        """
        self.num_inference_steps = num_inference_steps

        if shift is not None:
            self.shift = shift

        # Calculate denoising step list based on num_inference_steps
        # For 4 steps: [1000, 750, 500, 250]
        # For n steps: evenly spaced from num_train_timesteps to near 0
        step_size = self.num_train_timesteps / num_inference_steps
        self.denoising_step_list = []
        for i in range(num_inference_steps):
            step_value = self.num_train_timesteps - (i * step_size)
            self.denoising_step_list.append(int(step_value))

        # Generate full sigma schedule
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        all_sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]

        # Apply shift transformation
        all_sigmas = self.shift * all_sigmas / (1 + (self.shift - 1) * all_sigmas)
        all_timesteps = all_sigmas * self.num_train_timesteps

        # Select timesteps based on denoising_step_list
        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = all_timesteps[self.denoising_step_index].to(device)
        self.sigmas = all_sigmas[self.denoising_step_index].to("cpu")

        # Reset step tracking
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        """Current step index in the denoising process"""
        return self._step_index

    def _init_step_index(self, timestep):
        """Initialize step index based on current timestep"""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        # Find the index of the current timestep
        for i, t in enumerate(self.timesteps):
            if abs(t.item() - timestep) < 1e-5:
                self._step_index = i
                break
        else:
            # If not found, assume we're at the beginning
            self._step_index = 0

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Add noise to samples using the flow matching formula.

        Args:
            original_samples: Clean samples
            noise: Random noise
            sigma: Noise level

        Returns:
            Noisy samples
        """
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Perform one denoising step.

        Args:
            model_output: Predicted noise/velocity from the model
            timestep: Current timestep
            sample: Current noisy sample
            generator: Optional random generator for stochastic sampling

        Returns:
            Tuple containing the denoised sample for the next step
        """
        # Initialize step index if needed
        if self._step_index is None:
            self._init_step_index(timestep)

        # Convert to float32 for computation
        flow_pred = model_output.to(torch.float32)
        sample = sample.to(torch.float32)

        # Get current sigma
        sigma = self.sigmas[self._step_index].item()

        # Compute denoised sample using flow matching
        noisy_sample = sample - sigma * flow_pred

        # Add noise for next step if not at the last step
        if self._step_index < self.num_inference_steps - 1:
            next_sigma = self.sigmas[self._step_index + 1].item()

            # Generate noise if needed
            if next_sigma > 0:
                device = sample.device
                noise = torch.randn(
                    noisy_sample.shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator
                )
                noisy_sample = self.add_noise(noisy_sample, noise, next_sigma)

        # Increment step index
        self._step_index += 1

        # Return as tuple for compatibility
        return (noisy_sample.to(sample.dtype),)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Scale the input sample for the model. For flow matching, no scaling is needed.

        Args:
            sample: Input sample
            timestep: Current timestep (unused for flow matching)

        Returns:
            Unmodified sample
        """
        return sample