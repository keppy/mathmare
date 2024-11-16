import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from dataclasses import dataclass
from transformers import CLIPVisionModel, AutoModelForCausalLM
from pathlib import Path
import yaml
from typing import Optional
from .layers import Block, CrossAttention

@dataclass
class ModelConfig:
    """Configuration for math visualization model, loaded from YAML with defaults."""
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # Pretrained model paths
        self.vision_model: str = "openai/clip-vit-large-patch14"
        self.text_model: str = "microsoft/phi-2"

        # Architecture defaults
        self.image_size: int = 224  # CLIP default
        self.patch_size: int = 14   # CLIP default
        self.in_channels: int = 3
        self.hidden_size: int = 2048
        self.intermediate_size: int = 8192
        self.num_attention_heads: int = 32
        self.num_hidden_layers: int = 24
        self.num_cross_layers: int = 8

        # RoPE settings
        self.partial_rotary_factor: float = 0.5
        self.rope_theta: float = 10000.0

        # Other settings
        self.max_text_len: int = 512
        self.dropout: float = 0.0
        self.bias: bool = True
        self.max_seq_len: int = 2048

        # Load from YAML if provided
        if config_path is not None:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
                # Update from YAML
                for k, v in config_dict['model'].items():
                    setattr(self, k, v)

        # Override with any provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

class EnhancedMathVisualModel(nn.Module):
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__()
        self.config = ModelConfig(config_path, **kwargs)

        # Load pretrained vision encoder (CLIP)
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            self.config.vision_model
        )

        # Load pretrained text model (Phi-2)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            self.config.text_model
        )

        # Vision projection with LayerNorm (Moondream style)
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(self.vision_encoder.config.hidden_size),
            nn.Linear(self.vision_encoder.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )

        # Time embedding for diffusion
        time_dim = self.config.hidden_size * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Cross-attention between vision and text features
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttention(self.config)
            for _ in range(self.config.num_cross_layers)
        ])

        # U-Net blocks
        self.down_blocks = nn.ModuleList([
            nn.ModuleList([Block(self.config, is_cross=True) for _ in range(2)])
            for _ in range(3)
        ])

        self.mid_block = Block(self.config, is_cross=True)

        self.up_blocks = nn.ModuleList([
            nn.ModuleList([Block(self.config, is_cross=True) for _ in range(2)])
            for _ in range(3)
        ])

        # Final processing
        self.final_ln = nn.LayerNorm(self.config.hidden_size)
        self.final_proj = nn.Conv2d(self.config.hidden_size, self.config.in_channels, 1)

        self.initialize_weights()
        self.freeze_pretrained()

    def initialize_weights(self):
        """Initialize new components with standard initialization."""
        def _init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        # Only initialize new components
        self.vision_projection.apply(_init)
        self.time_mlp.apply(_init)
        self.cross_attention_blocks.apply(_init)
        self.down_blocks.apply(_init)
        self.mid_block.apply(_init)
        self.up_blocks.apply(_init)
        self.final_ln.apply(_init)
        self.final_proj.apply(_init)

    def freeze_pretrained(self):
        """Freeze pretrained components for initial training."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self):
        """Unfreeze last few layers for fine-tuning."""
        for name, param in self.vision_encoder.named_parameters():
            if "layer.-1" in name or "layer.-2" in name:
                param.requires_grad = True
        for name, param in self.text_model.named_parameters():
            if "h.-1" in name or "h.-2" in name:
                param.requires_grad = True

    def pos_encoding(self, t, channels):
        """Positional encoding for timesteps."""
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, timesteps, tokens=None, image_cond=None):
        """
        Forward pass with diffusion and conditioning

        Args:
            x (torch.Tensor): Noised image, shape (B, C, H, W)
            timesteps (torch.Tensor): Diffusion timesteps, shape (B,)
            tokens (torch.Tensor, optional): Conditioning text tokens
            image_cond (torch.Tensor, optional): Conditioning image
        """
        # Get conditioning features
        if image_cond is not None:
            with torch.no_grad():
                vision_features = self.vision_encoder(image_cond).last_hidden_state
            vision_features = self.vision_projection(vision_features)

        if tokens is not None:
            with torch.no_grad():
                text_features = self.text_model.transformer.wte(tokens)

            # Apply cross attention between vision and text
            for block in self.cross_attention_blocks:
                text_features = block(text_features, vision_features)

            context = text_features
        else:
            context = vision_features if image_cond is not None else None

        # Time embedding
        t = self.pos_encoding(timesteps, self.config.hidden_size)
        t = self.time_mlp(t)

        # Process input image
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + t.unsqueeze(1)

        # Down blocks with skip connections
        skip_connections = []
        pos_offset = 0

        for blocks in self.down_blocks:
            for block in blocks:
                x = block(x, context=context, pos_offset=pos_offset)
                pos_offset += x.shape[1]
            skip_connections.append(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))
            x = F.avg_pool2d(x, 2)
            x = rearrange(x, 'b c h w -> b (h w) c')

        # Middle
        x = self.mid_block(x, context=context, pos_offset=pos_offset)

        # Up blocks with skip connections
        for blocks in self.up_blocks:
            x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = x + skip_connections.pop()
            for block in blocks:
                x = block(x, context=context, pos_offset=pos_offset)
                pos_offset += x.shape[1]

        # Final processing
        x = self.final_ln(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))
        x = self.final_proj(x)

        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay handling and fused implementation when available.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters that need weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Use fused Adam implementation if available
        fused_available = device_type == 'cuda' and torch.cuda.is_available()
        extra_args = dict(fused=True) if fused_available else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        return optimizer

    @staticmethod
    def from_pretrained(path: str):
        """Load model from pretrained weights."""
        config_path = Path(path) / "config.yaml"
        weights_path = Path(path) / "model.pt"

        model = EnhancedMathVisualModel(config_path)
        model.load_state_dict(torch.load(weights_path))
        return model

    def save_pretrained(self, path: str):
        """Save model and config."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(save_path / "config.yaml", 'w') as f:
            yaml.dump({'model': self.config.__dict__}, f)

        # Save weights
        torch.save(self.state_dict(), save_path / "model.pt")
