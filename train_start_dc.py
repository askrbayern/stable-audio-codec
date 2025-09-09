#!/usr/bin/env python3
import os
import json
import argparse
# temporary
import time
# temporary
import glob
import itertools

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
    try:
        # Try loading with torchaudio first
        audio, sr = torchaudio.load(path)
    except Exception as e:
        print(f"Warning: torchaudio failed to load {path}: {e}")
        # Fallback to other backends for MP3
        try:
            import soundfile as sf
            audio, sr = sf.read(path, dtype='float32')
            audio = torch.from_numpy(audio).T  # soundfile returns (frames, channels)
        except Exception as e2:
            print(f"Warning: soundfile also failed: {e2}")
            try:
                import librosa
                audio, sr = librosa.load(path, sr=None, mono=False)
                audio = torch.from_numpy(audio)
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
            except Exception as e3:
                raise RuntimeError(f"All audio loading methods failed for {path}: torchaudio={e}, soundfile={e2}, librosa={e3}")
    
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    return audio


def save_audio(audio: torch.Tensor, path: str, sample_rate: int):
    audio = audio.squeeze(0)
    if torch.isnan(audio).any() or torch.isinf(audio).any():
        print("Warning: NaN or Inf detected in audio, replacing with zeros")
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = torch.clamp(audio, -0.99, 0.99)
    max_val = audio.abs().max()
    if max_val > 0.99:
        print(f"Warning: Audio max value {max_val:.3f} > 0.99, normalizing")
        audio = audio * 0.99 / max_val
    if audio.abs().max() < 1e-6:
        print(f"Warning: Audio is nearly silent (max: {audio.abs().max():.6f})")
        audio = torch.zeros_like(audio)

    torchaudio.save(path, audio, sample_rate)


def _dir_has_audio(path: str) -> bool:

    # Fast path: precomputed filelist
    flist = os.path.join(path, "filelist.txt")

    if os.path.isfile(flist):
        with open(flist, "r") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    return True
        # If here, filelist exists but appears empty
        return False

    return False


def assert_audio_exists_in_data_config(data_cfg: dict):
    datasets = data_cfg.get("datasets", [])
    if not datasets:
        raise ValueError("data_config has no 'datasets' entries")
    problems = []
    for ds in datasets:
        ds_path = ds.get("path")
        ds_id = ds.get("id", "<unknown>")
        if not ds_path or not isinstance(ds_path, str):
            problems.append(f"[{ds_id}] missing or invalid 'path'")
            continue
        if not os.path.isdir(ds_path):
            problems.append(f"[{ds_id}] not a directory: {ds_path}")
            continue
        if not _dir_has_audio(ds_path):
            problems.append(f"[{ds_id}] no audio files found under: {ds_path}")
    if problems:
        msg = "\n - ".join(["Dataset path/audio validation failed:"] + problems)
        raise ValueError(msg)


def recon(autoencoder, recon_path: str, output_path: str, device: torch.device, sample_rate: int):
    print(f"[Recon] {recon_path} -> {output_path}")
    if not output_path or output_path == "/dev/null":
        print(f"[Recon] Skipping reconstruction (output_path: {output_path})")
        return
    autoencoder = autoencoder.to(device)
    was_training = autoencoder.training
    autoencoder.eval()
    x = load_audio(recon_path, sample_rate).to(device)
    try:
        with torch.no_grad():
            latents, _ = autoencoder.encode(x, return_info=True)
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"Warning: NaN/Inf in latents, skipping reconstruction for {recon_path}")
                return
            y = autoencoder.decode(latents)
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"Warning: NaN/Inf in decoded output, attempting to fix")
                y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        y = y.clamp(-1, 1).cpu()
        save_audio(y, output_path, sample_rate)
    finally:
        if was_training:
            autoencoder.train()


class EpochReconCallback(pl.Callback):
    def __init__(self, recon_dir: str, output_dir: str, sample_rate: int, device: torch.device, every_n_epochs: int = 10):
        os.makedirs(output_dir, exist_ok=True)
        patterns = ["**/*.wav", "**/*.flac", "**/*.mp3"]
        audio_paths = []
        for pat in patterns:
            audio_paths.extend(glob.glob(os.path.join(recon_dir, pat), recursive=True))
        self.audio_paths = sorted(audio_paths)
        if not self.audio_paths:
            raise ValueError(f"No audio files found in {recon_dir}")
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.device = device
        self.every_n_epochs = max(1, int(every_n_epochs))

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return
        for recon_path in self.audio_paths:
            base = os.path.splitext(os.path.basename(recon_path))[0]
            output_path = os.path.join(self.output_dir, f"{base}_epoch_{epoch}.wav")
            try:
                recon(pl_module.autoencoder, recon_path, output_path, self.device, self.sample_rate)
            except Exception as e:
                print(f"Error during reconstruction of {recon_path}: {e}")
                print("Continuing training despite reconstruction error...")
        # Ensure module is back in training mode
        pl_module.train()


def audit_dataloader_balance(train_dl, audit_batches: int = 3):
    """Print per-batch domain counts and whether a full-band item is present.
    Assumes the dataset exposes dataset.get_info(i) returning a dict with keys
    like 'domain' and 'is_full_band'. Falls back gracefully if unavailable.
    """
    try:
        dataset = getattr(train_dl, 'dataset', None)
        batch_sampler = getattr(train_dl, 'batch_sampler', None)
        if dataset is None or batch_sampler is None:
            print("[BalanceAudit] Skipped: dataloader missing dataset or batch_sampler")
            return
        if not hasattr(dataset, 'get_info'):
            print("[BalanceAudit] Skipped: dataset has no get_info(i) to query metadata")
            return
        batches = list(itertools.islice(iter(batch_sampler), int(audit_batches)))
        if not batches:
            print("[BalanceAudit] No batches to audit")
            return
        for bi, idxs in enumerate(batches, 1):
            domains = []
            full_flags = []
            for i in idxs:
                try:
                    info = dataset.get_info(i)
                    domains.append(str(info.get('domain')))
                    full_flags.append(bool(info.get('is_full_band', False)))
                except Exception as e:
                    print(f"[BalanceAudit] get_info failed for index {i}: {e}")
                    domains.append('unknown')
                    full_flags.append(False)
            # Aggregate
            total = len(idxs)
            by_domain = {}
            for d in domains:
                by_domain[d] = by_domain.get(d, 0) + 1
            has_full = any(full_flags)
            print(f"[BalanceAudit] Batch {bi}: total={total}, domains={by_domain}, has_full_band={has_full}")
            if not has_full:
                print("[BalanceAudit] WARNING: batch lacks full-band item")
    except Exception as e:
        print(f"[BalanceAudit] Audit failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train with dataset JSON, epoch checkpoints, and recon")
    parser.add_argument("--model-config", required=True, help="Model configuration JSON")
    parser.add_argument("--data-config", required=True, help="Dataset configuration JSON for dataloader")
    parser.add_argument("--recon-dir", required=True, help="Reconstruction audio directory for reconstruction")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs and checkpoints")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--recon-every-epochs", type=int, default=10, help="Reconstruct every N epochs")
    parser.add_argument("--wandb-project", type=str, default="stable_audio_tools")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--drop-lm", action="store_true", help="If set with --ckpt-path, drop LM weights and states from checkpoint before resuming")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # temporary
    t0 = time.time()
    # temporary
    
    with open(args.model_config) as f:
        model_cfg = json.load(f)

    # temporary
    print(f"[Stage] Loaded model config in {time.time()-t0:.2f}s", flush=True)
    # temporary

    # W&B
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            resume="allow",
            id=args.wandb_run_name,
            save_dir=args.output_dir,
        )
    else:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_name,
            save_dir=args.output_dir,
        )

    # Dataset config
    t1 = time.time()
    with open(args.data_config) as f:
        data_cfg = json.load(f)
    print(f"[Stage] Loaded data config in {time.time()-t1:.2f}s", flush=True)

    # Fast-fail on wrong paths / empty audio dirs
    print("[Stage] Validating dataset roots...", flush=True)
    t2 = time.time()
    assert_audio_exists_in_data_config(data_cfg)
    print(f"[Stage] Dataset roots validated in {time.time()-t2:.2f}s", flush=True)

    sample_rate = model_cfg["sample_rate"]

    t3 = time.time()
    train_dl = create_dataloader_from_config(
        data_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=sample_rate,
        sample_size=model_cfg["sample_size"],
        audio_channels=model_cfg.get("audio_channels", 2),
    )
    print(f"[Stage] Dataloader created in {time.time()-t3:.2f}s", flush=True)

    # Optional audit of batch balance/full-band presence. Enable via env AUDIT_BALANCE=1
    if os.environ.get('AUDIT_BALANCE', '0') == '1':
        audit_batches = int(os.environ.get('AUDIT_BALANCE_BATCHES', '3'))
        audit_dataloader_balance(train_dl, audit_batches=audit_batches)
        # Recreate dataloader to avoid consuming sampler state
        train_dl = create_dataloader_from_config(
            data_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=sample_rate,
            sample_size=model_cfg["sample_size"],
            audio_channels=model_cfg.get("audio_channels", 2),
        )

    model = create_model_from_config(model_cfg)
    wrapper = create_training_wrapper_from_config(model_cfg, model)

    device = torch.device("cuda")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # If resuming but dropping LM, sanitize checkpoint by removing LM weights and LM-specific states
    ckpt_path_to_use = args.ckpt_path if args.ckpt_path else None
    weights_only_resume = False
    if args.drop_lm and ckpt_path_to_use and os.path.exists(ckpt_path_to_use):
        print(f"[Stage] drop-lm enabled: sanitizing checkpoint {ckpt_path_to_use}")
        ckpt = torch.load(ckpt_path_to_use, map_location="cpu")
        # Strip lm.* from main state_dict and DO NOT inject any LM weights
        # so that the current model's LM remains randomly initialized.
        sd = ckpt.get("state_dict", {})
        if isinstance(sd, dict):
            lm_old_keys = [k for k in sd.keys() if k.startswith("lm.")]
            if lm_old_keys:
                print(f"[Stage] Removing {len(lm_old_keys)} 'lm.*' weights from checkpoint state_dict (LM will stay random)")
            sd_clean = {k: v for k, v in sd.items() if not k.startswith("lm.")}
            ckpt["state_dict"] = sd_clean
        # Remove any separately saved LM dicts/metadata so on_load_checkpoint treats LM as fresh
        if "lm_state_dict" in ckpt:
            print("[Stage] Removing 'lm_state_dict' from checkpoint")
            ckpt.pop("lm_state_dict", None)
        if "lm_param_names" in ckpt:
            ckpt.pop("lm_param_names", None)
        # Optimizer/scheduler states may reference old LM params — drop them to avoid mismatch
        if "optimizer_states" in ckpt:
            print("[Stage] Dropping optimizer_states to avoid LM optimizer mismatch")
            ckpt.pop("optimizer_states", None)
        if "lr_schedulers" in ckpt:
            ckpt.pop("lr_schedulers", None)

        sanitized_path = os.path.join(checkpoint_dir, "last_dropLM.ckpt")
        torch.save(ckpt, sanitized_path)
        ckpt_path_to_use = sanitized_path
        print(f"[Stage] Wrote sanitized checkpoint to {sanitized_path}")
        # We will load weights manually to avoid optimizer/scheduler restore
        weights_only_resume = True

    # Save best-by-metric
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=True,
    )

    # Save periodic every N epochs (unconditional), and keep last.ckpt
    checkpoint_callback_periodic = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_epochs=max(1, int(args.checkpoint_every_epochs)),
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=True,
        filename="{epoch:04d}",
    )

    recon_callback = EpochReconCallback(
        recon_dir=args.recon_dir,
        output_dir=args.output_dir,
        sample_rate=sample_rate,
        device=device,
        every_n_epochs=args.recon_every_epochs,
    )

    # If we created a sanitized checkpoint and intend to avoid restoring optimizers/schedulers,
    # load weights manually and clear ckpt_path_to_use so Lightning won't try to restore trainer state.
    if weights_only_resume and ckpt_path_to_use and os.path.exists(ckpt_path_to_use):
        try:
            print(f"[Stage] Loading weights only from {ckpt_path_to_use} (no optimizer/scheduler restore)")
            ckpt = torch.load(ckpt_path_to_use, map_location="cpu")
            state_dict = ckpt.get("state_dict", {})
            missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Stage] Missing keys while loading: {len(missing)} (ok) e.g. {missing[:5]}")
            if unexpected:
                print(f"[Stage] Unexpected keys while loading: {len(unexpected)} e.g. {unexpected[:5]}")
            # Restore EMA if present
            if getattr(wrapper, 'use_ema', False) and getattr(wrapper, 'autoencoder_ema', None) is not None:
                if 'autoencoder_ema_state_dict' in ckpt:
                    wrapper.autoencoder_ema.ema_model.load_state_dict(ckpt['autoencoder_ema_state_dict'])
                if 'autoencoder_ema_object_state' in ckpt:
                    wrapper.autoencoder_ema.load_state_dict(ckpt['autoencoder_ema_object_state'])
            # Prevent PL from restoring training state
            ckpt_path_to_use = None
        except Exception as e:
            print(f"[Stage] Weights-only load failed: {e}. Proceeding without resume.")
            ckpt_path_to_use = None

    t4 = time.time()
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true",
        # ============ rank-aware: prevent Lightning from injecting its own distributed sampler ============
        use_distributed_sampler=False,
        # ==================================================================================================
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback_best, checkpoint_callback_periodic, recon_callback],
        enable_checkpointing=True,
        default_root_dir=args.output_dir,
    )
    print(f"[Stage] Trainer created in {time.time()-t4:.2f}s", flush=True)

    trainer.fit(wrapper, train_dl, ckpt_path=ckpt_path_to_use)


if __name__ == "__main__":
    main()

