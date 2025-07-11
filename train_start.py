# file: train_server.py
#!/usr/bin/env python3
import os
import json
import argparse
import math
import glob

import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.training.factory import create_training_wrapper_from_config

def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    # [C, T] -> [1, C, T]
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    return audio

def save_audio(audio: torch.Tensor, path: str, sample_rate: int):
    # [1, C, T] -> [C, T]
    audio = audio.squeeze(0)
    torchaudio.save(path, audio, sample_rate)

def recon(autoencoder, input_path: str, output_path: str, device: torch.device, sample_rate: int):
    print(f"[Recon] {input_path} -> {output_path}")
    
    # Skip reconstruction if output_path is None, empty, or /dev/null
    if not output_path or output_path == "/dev/null":
        print(f"[Recon] Skipping reconstruction (output_path: {output_path})")
        return
    
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    x = load_audio(input_path, sample_rate).to(device)
    with torch.no_grad():
        latents, _ = autoencoder.encode(x, return_info=True)
        y = autoencoder.decode(latents)
    y = y.clamp(-1, 1).cpu()
    save_audio(y, output_path, sample_rate)

def laplace_cdf(x, expectation, scale):
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(-(shifted_x).abs() / scale)

class EpochReconCallback(pl.Callback):
    def __init__(self, input_dir: str, output_dir: str, sample_rate: int, device: torch.device):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.audio_paths = sorted(glob.glob(os.path.join(input_dir, "*.mp3")))
        if not self.audio_paths:
            raise ValueError(f"No audio files found in {input_dir}")
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.device = device
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Reconstruct each audio file and save as <basename>_epoch_<N>.mp3
        epoch = trainer.current_epoch + 1
        for input_path in self.audio_paths:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(self.output_dir, f"{base}_epoch_{epoch}.mp3")
            recon(pl_module.autoencoder, input_path, output_path, self.device, self.sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Train with checkpoints and reconstruction after each epoch"
    )
    parser.add_argument("--model-config",     required=True,
                        help="Model configuration JSON with dithered_fsq + lm_config")
    parser.add_argument("--data-dir",         required=True,
                        help="Training data directory (audio_dir format)")
    parser.add_argument("--input-dir",      required=True,
                        help="Input audio directory for reconstruction")
    parser.add_argument("--output-dir",       required=True,
                        help="Directory to save reconstruction outputs and checkpoints")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--num-workers",  type=int, default=6)
    parser.add_argument("--max-epochs",   type=int, default=50)
    parser.add_argument("--precision",    type=str, default="16-mixed")
    parser.add_argument("--accelerator",  type=str, default="auto")
    parser.add_argument("--devices",      type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="stable_audio_tools", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--ckpt-path", type=str, default=None, help="resume from this checkpoint")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load config
    with open(args.model_config) as f:
        model_cfg = json.load(f)

    # 1.5 Setup W&B logger
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name)

    # 2. Construct dataset_config for DataLoader
    data_cfg = {
        "dataset_type": "audio_dir",
        "datasets": [{"id": "train", "path": args.data_dir}],
        "random_crop": True,
        "drop_last": True
    }
    sample_rate = model_cfg["sample_rate"]

    # 3. DataLoader
    train_dl = create_dataloader_from_config(
        data_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=sample_rate,
        sample_size=model_cfg["sample_size"],
        audio_channels=model_cfg.get("audio_channels", 2),
    )

    # 4. Model + Wrapper
    model = create_model_from_config(model_cfg)                      # AudioAutoencoder
    wrapper = create_training_wrapper_from_config(model_cfg, model)  # LightningModule
    lm_cfg = model_cfg["model"].get("lm", None)
    lm_weight = model_cfg["model"].get("lm_weight", 1.0)
    if lm_cfg is not None:
        from stable_audio_tools.models.lm_continuous import LaplaceLanguageModel
        wrapper.lm = LaplaceLanguageModel(wrapper.autoencoder.latent_dim, lm_cfg)
        wrapper.lm_weight = lm_weight
        wrapper.lm_config = lm_cfg

    device = torch.device("cuda" if torch.cuda.is_available() and args.devices>0 else "cpu")

    # 5. Setup callbacks
    # Checkpoint callback - save only the best performing checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="best_checkpoint",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=False
    )
    
    # Reconstruction callback - recon after every epoch
    recon_callback = EpochReconCallback(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=sample_rate,
        device=device
    )

    # 6. Training
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true",
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, recon_callback]
    )
    trainer.fit(wrapper, train_dl, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()