"""Training loop (Stage 5): joint weighted loss (or plain CE for the
baseline), Adam, Xavier init (applied at model construction), gradient
clipping, checkpoint save/resume, early stopping on dev BLEU (Section 2.3/2.6)."""
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.datasets.tokenizer import LIDTokenizer, WordTokenizer
from src.losses.attention_loss import AttentionCELoss
from src.losses.joint_loss import JointLoss
from src.models.baseline_transformer import BaselineTransformer
from src.models.layers import generate_causal_mask, key_padding_mask_from_lengths
from src.models.sign2_lid_text import Sign2LIDText
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.schedulers import build_gradient_clipper, build_optimizer, build_scheduler
from src.utils.logging import make_logger


def prepare_batch(batch: dict, tokenizer: WordTokenizer, device: str) -> dict:
    text_ids = batch["text_ids"].to(device)          # [B, |T|], ends with <eos>
    text_lengths = batch["text_lengths"].to(device)
    features = batch["features"].to(device)
    feature_lengths = batch["feature_lengths"].to(device)

    bsz, t_len = text_ids.shape
    bos_col = torch.full((bsz, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    tgt_input_ids = torch.cat([bos_col, text_ids[:, :-1]], dim=1)  # teacher forcing shift

    tgt_mask = generate_causal_mask(t_len, device=device)
    tgt_key_padding_mask = key_padding_mask_from_lengths(text_lengths, t_len)
    src_key_padding_mask = key_padding_mask_from_lengths(feature_lengths, features.size(1))

    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "src_key_padding_mask": src_key_padding_mask,
        "text_ids": text_ids,
        "text_lengths": text_lengths,
        "tgt_input_ids": tgt_input_ids,
        "tgt_mask": tgt_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask,
        "src_langs": batch["src_langs"],
        "tgt_langs": batch["tgt_langs"],
    }


class Trainer:
    def __init__(
        self,
        config: dict,
        model,
        txt_tokenizer: WordTokenizer,
        lid_tokenizer: Optional[LIDTokenizer] = None,
        family_router=None,
        model_dir: str = "experiments/results/run",
        device: str = "cpu",
    ):
        self.config = config
        self.model = model.to(device)
        self.txt_tokenizer = txt_tokenizer
        self.lid_tokenizer = lid_tokenizer
        self.family_router = family_router
        self.device = device
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.logger = make_logger(model_dir)

        self.is_baseline = isinstance(model, BaselineTransformer)
        if self.is_baseline:
            self.loss_fn = AttentionCELoss(
                pad_index=txt_tokenizer.pad_id,
                label_smoothing=config["loss"]["label_smoothing"],
            )
        else:
            self.loss_fn = JointLoss(
                lambda_lid=config["loss"]["lambda_lid"],
                lambda_txt=config["loss"]["lambda_txt"],
                lambda_attn=config["loss"]["lambda_attn"],
                txt_pad_index=txt_tokenizer.pad_id,
                label_smoothing=config["loss"]["label_smoothing"],
            )

        self.optimizer = build_optimizer(config, self.model.parameters())
        self.scheduler, self.scheduler_step_at = build_scheduler(config, self.optimizer)
        self.clip_grad = build_gradient_clipper(config)
        self.grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
        self._micro_step = 0

        self.epoch = 0
        self.step = 0
        self.best_score = -float("inf")
        self.epochs_without_improvement = 0

    def _lid_targets(self, batch: dict) -> torch.Tensor:
        assert self.lid_tokenizer is not None
        bsz, t_len = batch["text_ids"].shape
        rows = []
        for i in range(bsz):
            lang = batch["src_langs"][i]
            n = int(batch["text_lengths"][i])
            ids = self.lid_tokenizer.encode_repeated(lang, n)
            ids += [self.lid_tokenizer.pad_id] * (t_len - n)
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def training_step(self, batch: dict) -> dict:
        self.model.train()
        prepared = prepare_batch(batch, self.txt_tokenizer, self.device)

        if self.family_router is not None:
            self.family_router.set_batch_languages(prepared["src_langs"])

        if self._micro_step % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        if self.is_baseline:
            attn_log_probs = self.model(
                features=prepared["features"],
                src_key_padding_mask=prepared["src_key_padding_mask"],
                tgt_input_ids=prepared["tgt_input_ids"],
                tgt_mask=prepared["tgt_mask"],
                tgt_key_padding_mask=prepared["tgt_key_padding_mask"],
            )
            loss = self.loss_fn(attn_log_probs, prepared["text_ids"]) / prepared["text_ids"].size(0)
            losses = {"attn": loss, "total": loss}
        else:
            out = self.model(
                features=prepared["features"],
                src_key_padding_mask=prepared["src_key_padding_mask"],
                tgt_input_ids=prepared["tgt_input_ids"],
                tgt_mask=prepared["tgt_mask"],
                tgt_key_padding_mask=prepared["tgt_key_padding_mask"],
            )
            lid_ids = self._lid_targets(batch) if self.model.use_lid_branch else None
            losses = self.loss_fn(
                lid_log_probs=out["lid_log_probs"],
                txt_ctc_log_probs=out["txt_ctc_log_probs"],
                attn_log_probs=out["attn_log_probs"],
                feature_lengths=prepared["feature_lengths"],
                text_ids=prepared["text_ids"],
                text_lengths=prepared["text_lengths"],
                lid_ids=lid_ids,
            )
            loss = losses["total"]

        (loss / self.grad_accum_steps).backward()
        self._micro_step += 1
        if self._micro_step % self.grad_accum_steps == 0:
            self._optimizer_step()
        return {k: float(v.detach()) for k, v in losses.items()}

    def _optimizer_step(self) -> None:
        if self.clip_grad is not None:
            self.clip_grad(self.model.parameters())
        self.optimizer.step()
        self.step += 1

    def _flush_accumulated_grads(self) -> None:
        # apply a partial accumulation window left over at epoch end
        if self._micro_step % self.grad_accum_steps != 0:
            self._optimizer_step()
            self._micro_step = 0

    def save(self, tag: str) -> str:
        path = os.path.join(self.model_dir, f"{tag}.ckpt")
        save_checkpoint(path, self.model, self.optimizer, self.scheduler, self.epoch, self.step, self.best_score)
        return path

    def resume(self, path: str) -> None:
        state = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        self.epoch = state["epoch"]
        self.step = state["step"]
        self.best_score = state["best_score"]

    def fit(self, train_loader: DataLoader, dev_eval_fn=None) -> None:
        """dev_eval_fn(model) -> dev_bleu, called once per epoch for early
        stopping / plateau scheduling. Passing None just trains for
        max_epochs with no early stopping (used by unit tests)."""
        max_epochs = self.config["training"]["max_epochs"]
        patience = self.config["training"]["early_stopping_patience"]
        last_checkpoint_time = time.time()
        checkpoint_every_minutes = self.config["training"].get("checkpoint_every_minutes")

        epoch_iter = range(max_epochs) if max_epochs else iter(int, 1)  # infinite if None
        for _ in epoch_iter:
            self.epoch += 1
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                losses = self.training_step(batch)
                epoch_loss += losses["total"]
                n_batches += 1
            self._flush_accumulated_grads()
            self.logger.info("epoch %d: mean total loss %.4f", self.epoch, epoch_loss / max(n_batches, 1))
            conditioner = getattr(self.model, "lid_conditioner", None)
            if conditioner is not None:
                self.logger.info("epoch %d: lid self-conditioning gate g=%.4f", self.epoch, conditioner.gate.item())

            if self.epoch % self.config["training"]["checkpoint_every_epochs"] == 0:
                self.save("latest")
            if checkpoint_every_minutes and time.time() - last_checkpoint_time > checkpoint_every_minutes * 60:
                self.save("latest")
                last_checkpoint_time = time.time()

            if dev_eval_fn is not None:
                dev_bleu = dev_eval_fn(self.model)
                self.logger.info("epoch %d: dev BLEU %.2f (best so far %.2f)", self.epoch, dev_bleu, max(dev_bleu, self.best_score))
                if self.scheduler_step_at == "validation":
                    self.scheduler.step(dev_bleu)
                if dev_bleu > self.best_score:
                    self.best_score = dev_bleu
                    self.epochs_without_improvement = 0
                    self.save("best")
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= patience:
                        self.logger.info("early stopping at epoch %d (no dev-BLEU improvement for %d epochs)", self.epoch, patience)
                        break
            elif self.scheduler_step_at == "epoch":
                self.scheduler.step()
