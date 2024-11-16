"""
Training script for mathematical visualization model.
Supports progressive training phases and diffusion scheduling.
"""

import os
import time
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import yaml
from contextlib import nullcontext
import argparse
from typing import Optional
import numpy as np

from model.model import EnhancedMathVisualModel
from data.dataset import MathDataset

def setup_logging(output_dir: Path):
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s',
                       handlers=[
                           logging.FileHandler(output_dir / 'train.log'),
                           logging.StreamHandler()
                       ])

class DiffusionScheduler:
    """Manages noise scheduling for diffusion process"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        """Add noise to image at timestep t"""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])

        noise = torch.randn_like(x)
        return (
            sqrt_alphas_cumprod.view(-1, 1, 1, 1) * x +
            sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1) * noise,
            noise
        )

class TrainingPhase:
    """Manages progressive training phases"""
    def __init__(self, config: dict):
        self.phases = [
            {
                "name": "new_components",
                "epochs": config['phase1_epochs'],
                "lr": config['phase1_lr'],
                "unfreeze": ["vision_projection", "cross_attention_blocks",
                            "down_blocks", "up_blocks", "final_proj"]
            },
            {
                "name": "partial_finetune",
                "epochs": config['phase2_epochs'],
                "lr": config['phase2_lr'],
                "unfreeze": ["vision_encoder.encoder.layers.-1",
                            "vision_encoder.encoder.layers.-2"]
            },
            {
                "name": "full_finetune",
                "epochs": config['phase3_epochs'],
                "lr": config['phase3_lr'],
                "unfreeze": "all"
            }
        ]
        self.current_phase = 0

    def get_current_phase(self):
        return self.phases[self.current_phase]

    def next_phase(self):
        self.current_phase += 1
        return self.current_phase < len(self.phases)

    def apply_phase(self, model: EnhancedMathVisualModel):
        phase = self.get_current_phase()
        if phase["unfreeze"] == "all":
            model.unfreeze_all()
        else:
            model.freeze_pretrained()
            for name in phase["unfreeze"]:
                model.unfreeze_layer(name)
        return phase["lr"]

def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    diffusion: DiffusionScheduler,
    config: dict,
    ctx: nullcontext,
    epoch: int,
    device: str,
    rank: int
):
    """Train for one epoch with diffusion"""
    model.train()
    losses = []
    t0 = time.time()

    for it, batch in enumerate(train_loader):
        # Get learning rate
        lr = get_lr(it, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass with mixed precision
        with ctx:
            # Get clean images and condition
            clean_images = batch['image'].to(device)
            tokens = batch['tokens'].to(device) if 'tokens' in batch else None
            condition_images = batch.get('condition_image', None)
            if condition_images is not None:
                condition_images = condition_images.to(device)

            # Sample timesteps and add noise
            timesteps = torch.randint(
                0, diffusion.num_timesteps,
                (clean_images.shape[0],),
                device=device
            )
            noisy_images, noise = diffusion.add_noise(clean_images, timesteps)

            # Model prediction
            pred_noise = model(
                x=noisy_images,
                timesteps=timesteps,
                tokens=tokens,
                image_cond=condition_images
            )

            # Compute loss
            loss = F.mse_loss(pred_noise, noise)
            loss = loss / config['gradient_accumulation_steps']

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation and update
        if (it + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        # Logging
        if rank == 0 and it % config['log_interval'] == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            logging.info(
                f"epoch {epoch} iter {it}: loss {loss.item():.4f}, "
                f"time {dt*1000:.2f}ms, lr {lr:.2e}"
            )

def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup environment
    device, rank, world_size, ctx = setup_training(config)

    # Initialize model and training components
    model = EnhancedMathVisualModel(config_path=args.config)
    diffusion = DiffusionScheduler(**config['diffusion'])
    training_phases = TrainingPhase(config)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.to(device)

    # Training loop with phases
    while True:
        # Get current phase settings
        lr = training_phases.apply_phase(model)
        phase = training_phases.get_current_phase()

        # Configure optimizer for current phase
        optimizer = model.configure_optimizers(
            weight_decay=config['weight_decay'],
            learning_rate=lr,
            betas=tuple(config['betas']),
            device_type='cuda'
        )
        scaler = torch.cuda.amp.GradScaler(enabled=config['dtype'] == 'float16')

        # Create dataloaders
        train_loader, val_loader, train_sampler = create_dataloaders(
            config, world_size, rank
        )

        # Train for phase epochs
        for epoch in range(phase['epochs']):
            if world_size > 1:
                train_sampler.set_epoch(epoch)

            train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=None,
                scaler=scaler,
                diffusion=diffusion,
                config=config,
                ctx=ctx,
                epoch=epoch,
                device=device,
                rank=rank
            )

            # Save checkpoint
            if rank == 0 and epoch % config['save_interval'] == 0:
                save_checkpoint(
                    model, optimizer, config, epoch,
                    Path(args.out_dir) / f"phase_{phase['name']}_epoch_{epoch}.pt"
                )

        # Move to next phase or end training
        if not training_phases.next_phase():
            break

if __name__ == '__main__':
    main()
