"""Master World Model Trainer and Overfit Diagnostic Pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prism.world_model.config import WorldModelConfig
from prism.world_model.inputs import ModelInputs
from prism.world_model.model import CausalWorldModel
from prism.training.normalization import ObservationNormalizer
from prism.training.checkpointing import save_checkpoint
from prism.training.metrics import EvaluationSummary, compute_regression_metrics


@dataclass
class TrainingHistory:
    """Historical tracking of training and validation loss components."""

    train_total_loss: List[float] = field(default_factory=list)
    train_obs_loss: List[float] = field(default_factory=list)
    train_trans_loss: List[float] = field(default_factory=list)
    train_kl_loss: List[float] = field(default_factory=list)
    val_total_loss: List[float] = field(default_factory=list)
    val_mae: List[float] = field(default_factory=list)
    val_rmse: List[float] = field(default_factory=list)


class WorldModelTrainer:
    """Encapsulates end-to-end training, validation, early stopping, and overfit diagnostics."""

    def __init__(
        self,
        model: CausalWorldModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        normalizer: Optional[ObservationNormalizer] = None,
        config: Optional[WorldModelConfig] = None,
        device: Optional[torch.device] = None,
        save_dir: str | Path = "artifacts/baseline",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.config = config or WorldModelConfig()
        self.device = device or torch.device("cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.global_step = 0
        self.current_epoch = 0
        self.history = TrainingHistory()

    def train_epoch(self) -> Dict[str, float]:
        """Execute one complete training epoch across batches."""
        self.model.train()
        total_loss_acc = 0.0
        obs_loss_acc = 0.0
        trans_loss_acc = 0.0
        kl_loss_acc = 0.0
        num_batches = 0

        for batch in self.train_loader:
            inputs: ModelInputs = batch["inputs"]
            # Move tensors to target device
            obs = inputs.observations.to(self.device)
            mask = inputs.observation_mask.to(self.device)
            act = inputs.actions.to(self.device)
            dev_inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)

            self.optimizer.zero_grad()
            loss_out = self.model.compute_loss(dev_inputs)
            loss_out.total_loss.backward()

            # Gradient clipping for stability
            if self.config.training.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)

            self.optimizer.step()

            total_loss_acc += loss_out.total_loss.item()
            obs_loss_acc += loss_out.obs_loss.item()
            trans_loss_acc += loss_out.trans_loss.item()
            kl_loss_acc += loss_out.kl_loss.item()
            num_batches += 1
            self.global_step += 1

        avg_metrics = {
            "loss/train_total": total_loss_acc / max(1, num_batches),
            "loss/train_obs": obs_loss_acc / max(1, num_batches),
            "loss/train_trans": trans_loss_acc / max(1, num_batches),
            "loss/train_kl": kl_loss_acc / max(1, num_batches),
        }

        self.history.train_total_loss.append(avg_metrics["loss/train_total"])
        self.history.train_obs_loss.append(avg_metrics["loss/train_obs"])
        self.history.train_trans_loss.append(avg_metrics["loss/train_trans"])
        self.history.train_kl_loss.append(avg_metrics["loss/train_kl"])

        return avg_metrics

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Tuple[float, EvaluationSummary]:
        """Evaluate one-step predictive performance in physical unnormalized units."""
        loader = dataloader or self.val_loader
        if loader is None:
            raise ValueError("No dataloader provided for evaluation")

        self.model.eval()
        val_loss_acc = 0.0
        num_batches = 0

        all_preds_denorm = []
        all_targets_raw = []
        all_masks = []
        all_logvar_denorm = []

        with torch.no_grad():
            for batch in loader:
                inputs: ModelInputs = batch["inputs"]
                raw_obs = batch["raw_observations"].to(self.device)   # [B, L, 8] unnormalized
                raw_mask = batch["raw_mask"].to(self.device)          # [B, L, 8]

                obs = inputs.observations.to(self.device)
                mask = inputs.observation_mask.to(self.device)
                act = inputs.actions.to(self.device)
                dev_inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)

                loss_out = self.model.compute_loss(dev_inputs)
                val_loss_acc += loss_out.total_loss.item()
                num_batches += 1

                # Evaluate one-step predictive accuracy at t+1:
                # 1. Encode history up to t: [B, T, d_z]
                post_latents, _ = self.model.encode(dev_inputs)
                
                # 2. Transition Z_t -> Z_{t+1} using actions at t
                z_curr = post_latents.mean[:, :-1]   # [B, T-1, d_z]
                act_curr = act[:, :-1]              # [B, T-1, 4]
                prior_trans = self.model.transition_step(z_curr, act_curr)

                # 3. Decode Z_{t+1} -> O_{t+1} prediction: [B, T-1, 8]
                pred_obs_dist = self.model.decode(prior_trans.mean)

                # 4. Denormalize predictions to physical sensor units
                if self.normalizer is not None:
                    pred_denorm = self.normalizer.denormalize(pred_obs_dist.mean.cpu()).to(self.device)
                    # Denormalize variance: var_phys = var_norm * (std_train)^2
                    std_t = self.normalizer._std_tensor.to(self.device)
                    var_denorm = pred_obs_dist.variance * (std_t ** 2)
                    logvar_denorm = torch.log(var_denorm + 1e-8)
                else:
                    pred_denorm = pred_obs_dist.mean
                    logvar_denorm = pred_obs_dist.logvar

                # Targets are actual observation at t+1
                target_denorm = raw_obs[:, 1:]
                target_mask = raw_mask[:, 1:]

                all_preds_denorm.append(pred_denorm.reshape(-1, 8))
                all_targets_raw.append(target_denorm.reshape(-1, 8))
                all_masks.append(target_mask.reshape(-1, 8))
                all_logvar_denorm.append(logvar_denorm.reshape(-1, 8))

        avg_val_loss = val_loss_acc / max(1, num_batches)
        
        flat_preds = torch.cat(all_preds_denorm, dim=0).cpu()
        flat_targets = torch.cat(all_targets_raw, dim=0).cpu()
        flat_masks = torch.cat(all_masks, dim=0).cpu()
        flat_logvar = torch.cat(all_logvar_denorm, dim=0).cpu()

        summary = compute_regression_metrics(
            predictions=flat_preds,
            targets=flat_targets,
            mask=flat_masks,
            pred_logvar=flat_logvar,
            model_name="PRISM_WorldModel",
        )

        return avg_val_loss, summary

    def fit(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Execute full training loop with early stopping and model checkpointing."""
        epochs = max_epochs or self.config.training.max_epochs
        best_val_mae = float("inf")
        patience_counter = 0

        print(f"Starting World Model training on {self.device} for {epochs} epochs...")

        for ep in range(epochs):
            self.current_epoch = ep + 1
            t0 = time.time()
            train_m = self.train_epoch()
            t_train = time.time() - t0

            val_loss, val_summary = self.evaluate(self.val_loader)
            self.history.val_total_loss.append(val_loss)
            self.history.val_mae.append(val_summary.overall_mae)
            self.history.val_rmse.append(val_summary.overall_rmse)

            # Check for best validation MAE
            is_best = val_summary.overall_mae < best_val_mae
            if is_best:
                best_val_mae = val_summary.overall_mae
                patience_counter = 0
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    val_loss=val_loss,
                    val_mae=val_summary.overall_mae,
                    config=self.config,
                    normalizer=self.normalizer,
                    save_path=self.save_dir / "best.pt",
                )
            else:
                patience_counter += 1

            # Save last checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                global_step=self.global_step,
                val_loss=val_loss,
                val_mae=val_summary.overall_mae,
                config=self.config,
                normalizer=self.normalizer,
                save_path=self.save_dir / "last.pt",
            )

            # Save normalization stats
            if self.normalizer is not None:
                self.normalizer.save_yaml(self.save_dir / "normalization.yaml")

            print(
                f"Epoch [{self.current_epoch:02d}/{epochs:02d}] ({t_train:.1f}s) | "
                f"Train Loss: {train_m['loss/train_total']:.3f} (Obs: {train_m['loss/train_obs']:.3f}, Trans: {train_m['loss/train_trans']:.3f}) | "
                f"Val MAE: {val_summary.overall_mae:.3f} | Best MAE: {best_val_mae:.3f}"
            )

            if patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping triggered at epoch {self.current_epoch} (patience={self.config.training.early_stopping_patience})")
                break

        return {
            "best_val_mae": best_val_mae,
            "final_epoch": self.current_epoch,
            "global_step": self.global_step,
        }

    def overfit_diagnostic(self, num_epochs: int = 50) -> float:
        """Critical Diagnostic: Deliberately overfit a tiny single-batch subset.
        
        Verifies that gradient flow, loss formulations, and architecture are capable of
        driving training error toward zero on a tiny dataset.
        """
        self.model.train()
        single_batch = next(iter(self.train_loader))
        inputs: ModelInputs = single_batch["inputs"]

        obs = inputs.observations.to(self.device)
        mask = inputs.observation_mask.to(self.device)
        act = inputs.actions.to(self.device)
        dev_inputs = ModelInputs(observations=obs, observation_mask=mask, actions=act)

        initial_loss = 0.0
        final_loss = 0.0

        for step in range(num_epochs):
            self.optimizer.zero_grad()
            loss_out = self.model.compute_loss(dev_inputs)
            loss_val = loss_out.total_loss.item()

            if step == 0:
                initial_loss = loss_val

            loss_out.total_loss.backward()
            self.optimizer.step()
            final_loss = loss_val

        return final_loss
