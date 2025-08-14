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
    
    # Check for NaN/Inf values to prevent MP3 encoding crashes
    if torch.isnan(audio).any() or torch.isinf(audio).any():
        print(f"Warning: NaN or Inf detected in audio, replacing with zeros")
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Ensure values are in valid range (slightly below 1.0 to be safe)
    audio = torch.clamp(audio, -0.99, 0.99)
    
    # Additional safety: normalize if values are too extreme
    max_val = audio.abs().max()
    if max_val > 0.99:
        print(f"Warning: Audio max value {max_val:.3f} > 0.99, normalizing")
        audio = audio * 0.99 / max_val
    
    # Check for very small values that might cause issues
    if audio.abs().max() < 1e-6:
        print(f"Warning: Audio is nearly silent (max: {audio.abs().max():.6f})")
        audio = torch.zeros_like(audio)
    
    # Save as WAV first to avoid MP3 encoding issues
    if path.endswith('.mp3'):
        # Save as WAV instead of MP3 to avoid encoder crashes
        wav_path = path.replace('.mp3', '.wav')
        print(f"Saving as WAV to avoid MP3 encoding issues: {wav_path}")
        torchaudio.save(wav_path, audio, sample_rate)
    else:
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
        
        # Check latents for issues
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"Warning: NaN/Inf in latents, skipping reconstruction for {input_path}")
            return
            
        y = autoencoder.decode(latents)
        
        # Check decoded output immediately
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"Warning: NaN/Inf in decoded output, attempting to fix")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            
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
        
        # Only reconstruct every 20 epochs
        if epoch % 10 != 0:
            return
            
        for input_path in self.audio_paths:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(self.output_dir, f"{base}_epoch_{epoch}.mp3")
            
            try:
                recon(pl_module.autoencoder, input_path, output_path, self.device, self.sample_rate)
            except Exception as e:
                print(f"Error during reconstruction of {input_path}: {e}")
                print("Continuing training despite reconstruction error...")
                # Try saving as WAV if MP3 failed
                try:
                    wav_path = output_path.replace('.mp3', '.wav')
                    recon(pl_module.autoencoder, input_path, wav_path, self.device, self.sample_rate)
                    print(f"Successfully saved as WAV: {wav_path}")
                except Exception as e2:
                    print(f"Also failed to save as WAV: {e2}")


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
    parser.add_argument("--inspect", action="store_true", help="Inspect mode: extract latents and save to output_dir/inspect")
    parser.add_argument("--inspect-count", type=int, default=10, help="Number of samples to extract latents for in inspect mode")
    parser.add_argument("--wandb-project", type=str, default="stable_audio_tools", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--ckpt-path", type=str, default=None, help="resume from this checkpoint")
    args = parser.parse_args()

    # Inspect mode: load raw audio files and extract latents
    if args.inspect:
        # Inspect mode: load raw audio files and extract latents
        device = torch.device("cuda" if torch.cuda.is_available() and args.devices>0 else "cpu")
        model_cfg_path = args.model_config
        with open(model_cfg_path) as f:
            model_cfg = json.load(f)
        # build model and load checkpoint
        autoencoder = create_model_from_config(model_cfg)
        # load weights from checkpoint if provided
        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location=device)
            # handle LightningModule checkpoint
            sd = ckpt.get('state_dict', ckpt)
            # filter and strip 'autoencoder.' prefix
            ae_sd = {k.replace('autoencoder.', ''): v for k, v in sd.items() if k.startswith('autoencoder.')}
            autoencoder.load_state_dict(ae_sd)
        autoencoder = autoencoder.to(device).eval()
        sample_rate = model_cfg.get("sample_rate")
        # Collect audio files from input_dir
        from glob import glob
        # recursively collect audio files under input_dir
        audio_patterns = ["**/*.wav", "**/*.mp3", "**/*.flac", "**/*.ogg"]
        audio_files = []
        for pat in audio_patterns:
            audio_files.extend(glob(os.path.join(args.input_dir, pat), recursive=True))
        audio_files = sorted(audio_files)[: args.inspect_count]
        # Create inspect output folder
        inspect_dir = os.path.join(args.output_dir, "inspect")
        os.makedirs(inspect_dir, exist_ok=True)
        for idx, path in enumerate(audio_files):
            # load full audio and cap to 1 minute
            raw = load_audio(path, sample_rate)  # [1, C, T]
            max_frames = sample_rate * 60
            if raw.shape[-1] > max_frames:
                raw = raw[..., :max_frames]
            x = raw.to(device)
            with torch.no_grad():
                latents, _ = autoencoder.encode(x, return_info=True)
            # save latents
            out_path = os.path.join(inspect_dir, f"latent_{idx}.pt")
            torch.save(latents.cpu(), out_path)
            print(f"Saved latents for {os.path.basename(path)} -> {out_path}")
            # reconstruct from latents
            with torch.no_grad():
                recon_audio = autoencoder.decode(latents)
            # clamp and move to CPU
            recon_audio = recon_audio.clamp(-1, 1).cpu()
            # save reconstructed audio with original extension
            base, ext = os.path.splitext(os.path.basename(path))
            recon_path = os.path.join(inspect_dir, f"{base}_recon{ext}")
            save_audio(recon_audio, recon_path, sample_rate)
            print(f"Saved reconstruction for {base} -> {recon_path}")
        return

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
        save_last=True
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