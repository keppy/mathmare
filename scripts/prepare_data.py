"""
Enhanced data preparation script supporting diffusion model training
and pretrained model requirements.
"""

import os
import json
import shutil
import hashlib
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, AutoTokenizer

@dataclass
class ProcessingConfig:
    # Image processing
    min_size: int = 512
    target_size: int = 224  # CLIP's native size
    max_aspect_ratio: float = 1.5
    num_workers: int = 4

    # Tokenization
    max_text_length: int = 512
    vision_model: str = "openai/clip-vit-large-patch14"
    text_model: str = "microsoft/phi-2"

    # Augmentation
    enable_augmentation: bool = True
    center_crop: bool = True
    random_flip: bool = True

    # Preprocessing
    normalize_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)  # CLIP values
    normalize_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)  # CLIP values

class ImageProcessor:
    """Handles image processing with CLIP-compatible transforms"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.clip_processor = CLIPProcessor.from_pretrained(config.vision_model)

        # Training transforms
        self.train_transforms = transforms.Compose([
            transforms.Resize(config.target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.target_size) if config.center_crop else transforms.RandomCrop(config.target_size),
            transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        ])

        # Validation transforms
        self.val_transforms = transforms.Compose([
            transforms.Resize(config.target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        ])

    def process_image(
        self,
        image_path: Path,
        output_dir: Path,
        is_training: bool = True
    ) -> Optional[str]:
        """Process single image with validation and transforms"""
        try:
            img = Image.open(image_path).convert('RGB')

            # Validate size
            if min(img.size) < self.config.min_size:
                logging.warning(f"{image_path} too small: {img.size}")
                return None

            # Validate aspect ratio
            aspect = max(img.size) / min(img.size)
            if aspect > self.config.max_aspect_ratio:
                logging.warning(f"{image_path} bad aspect ratio: {aspect:.2f}")
                return None

            # Apply transforms
            transform = self.train_transforms if is_training else self.val_transforms
            processed = transform(img)

            # Generate hash and save
            img_hash = hashlib.md5(processed.numpy().tobytes()).hexdigest()
            out_path = output_dir / f"{img_hash}.pt"
            torch.save(processed, out_path)

            return img_hash

        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None

class TextProcessor:
    """Handles text processing with pretrained tokenizers"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model)

    def process_text(self, text: str) -> Dict:
        """Process text with tokenizer"""
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt"
        )

def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    concept_file: Path,
    config: ProcessingConfig
):
    """Enhanced dataset processing"""
    # Setup directories and processors
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    image_processor = ImageProcessor(config)
    text_processor = TextProcessor(config)

    # Load and validate concepts
    concepts = validate_concepts(concept_file)

    # Load metadata
    with open(raw_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Process dataset
    processed_items = []
    image_hashes = set()

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = []

        # Submit image processing jobs
        for item in metadata:
            image_path = raw_dir / "images" / item["image_file"]
            future = executor.submit(
                image_processor.process_image,
                image_path,
                image_dir,
                True  # is_training
            )
            futures.append((future, item))

        # Process results and tokenize text
        for future, item in tqdm(futures, desc="Processing"):
            img_hash = future.result()
            if img_hash and img_hash not in image_hashes:
                image_hashes.add(img_hash)

                # Create rich text prompt
                prompt = f"""Title: {item['title']}
                Topic: {item['topic_area']}
                Concepts: {', '.join(item['concepts'])}
                Description: {item['description']}
                """

                # Add tokenized text
                tokenized = text_processor.process_text(prompt)

                # Update item with processed data
                processed_item = {
                    **item,
                    "image_file": f"{img_hash}.pt",
                    "input_ids": tokenized["input_ids"].tolist(),
                    "attention_mask": tokenized["attention_mask"].tolist()
                }

                processed_items.append(processed_item)

    # Create stratified splits
    splits = create_stratified_splits(
        processed_items,
        train_ratio=0.8,
        val_ratio=0.1,
        concepts=concepts
    )

    # Save everything
    for split_name, items in splits.items():
        with open(output_dir / f"{split_name}.json", 'w') as f:
            json.dump(items, f, indent=2)

    # Save configs
    shutil.copy2(concept_file, output_dir / "concepts.yaml")
    with open(output_dir / "processing_config.yaml", 'w') as f:
        yaml.dump(asdict(config), f)

    logging.info(f"Processed {len(processed_items)} items")
    for split_name, items in splits.items():
        logging.info(f"{split_name}: {len(items)} items")

def create_stratified_splits(
    items: List[Dict],
    train_ratio: float,
    val_ratio: float,
    concepts: Dict
) -> Dict[str, List[Dict]]:
    """Create stratified splits based on concepts and difficulty"""
    # Group items by concept and difficulty
    groups = {}
    for item in items:
        for concept in item['concepts']:
            key = (concept, item.get('difficulty', 0))
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

    # Create splits maintaining concept distribution
    train_items, val_items, test_items = [], [], []

    for group in groups.values():
        n = len(group)
        indices = torch.randperm(n)

        train_idx = indices[:int(n * train_ratio)]
        val_idx = indices[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
        test_idx = indices[int(n * (train_ratio + val_ratio)):]

        train_items.extend([group[i] for i in train_idx])
        val_items.extend([group[i] for i in val_idx])
        test_items.extend([group[i] for i in test_idx])

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--concept_file", type=Path, required=True)
    parser.add_argument("--config", type=str, help="Optional config file")
    args = parser.parse_args()

    # Load config
    config = ProcessingConfig()
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
            config = ProcessingConfig(**config_dict)

    setup_logging(args.output_dir)
    process_dataset(args.raw_dir, args.output_dir, args.concept_file, config)

if __name__ == "__main__":
    main()
