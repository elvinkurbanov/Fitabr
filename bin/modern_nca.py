import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import lib
from lib import KWArgs


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]
    model: KWArgs
    optimizer: KWArgs
    batch_size: int
    patience: int
    n_epochs: Union[int, float]


class MLPBlock(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, d_in),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ModernNCA(nn.Module):
    """ModernNCA adapted to Fitabr's numerical/binary/categorical input dict."""

    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        dim: int,
        dropout: float,
        d_block: int,
        n_blocks: int,
        num_embeddings: Optional[dict],
        temperature: float = 1.0,
        sample_rate: float = 0.8,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1]")

        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.d_out = 1 if n_classes is None else n_classes
        self.temperature = temperature
        self.sample_rate = sample_rate

        self.num_embeddings = (
            None
            if num_embeddings is None or n_num_features == 0
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )
        self.cat_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )

        num_width = n_num_features
        if self.num_embeddings is not None:
            num_width *= num_embeddings["d_embedding"]
        d_in = num_width + n_bin_features + sum(cat_cardinalities)
        if d_in == 0:
            raise ValueError("ModernNCA received a dataset with no input features")

        self.encoder = nn.Linear(d_in, dim)
        self.post_encoder = (
            nn.Sequential(
                *(MLPBlock(dim, d_block, dropout) for _ in range(n_blocks)),
                nn.BatchNorm1d(dim),
            )
            if n_blocks > 0
            else None
        )

    def _pack_features(self, x_: dict[str, Tensor]) -> Tensor:
        parts = []
        if "num" in x_:
            x_num = x_["num"]
            if self.num_embeddings is not None:
                x_num = self.num_embeddings(x_num).flatten(1)
            parts.append(x_num)
        if "bin" in x_:
            parts.append(x_["bin"])
        if "cat" in x_:
            if self.cat_encoder is None:
                raise RuntimeError("Categorical input was supplied without cardinalities")
            parts.append(self.cat_encoder(x_["cat"]))
        return torch.cat(parts, dim=-1).to(self.encoder.weight.dtype)

    def forward(
        self,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        is_train: bool,
    ) -> Tensor:
        if is_train and len(candidate_y) > 0:
            retrieval_size = max(1, int(len(candidate_y) * self.sample_rate))
            retrieval_size = min(retrieval_size, len(candidate_y))
            sample_idx = torch.randperm(
                len(candidate_y), device=candidate_y.device
            )[:retrieval_size]
            candidate_x_ = {
                feature_type: values[sample_idx]
                for feature_type, values in candidate_x_.items()
            }
            candidate_y = candidate_y[sample_idx]

        x = self._pack_features(x_)
        candidate_x = self._pack_features(candidate_x_)

        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)
        if self.post_encoder is not None:
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)

        if is_train:
            if y is None:
                raise ValueError("Training labels are required")
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])

        if self.n_classes is not None:
            candidate_y = F.one_hot(
                candidate_y.long(), self.d_out
            ).to(x.dtype)
        elif candidate_y.ndim == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        distances = torch.cdist(x, candidate_x, p=2) / self.temperature
        if is_train:
            # The query batch was prepended to the candidate set above, so the
            # diagonal consists exactly of self-neighbours and must be excluded.
            distances.fill_diagonal_(torch.inf)

        neighbor_weights = F.softmax(-distances, dim=-1)
        output = neighbor_weights @ candidate_y
        if self.n_classes is not None:
            output = torch.log(output + 1e-7)
        return output.squeeze(-1)


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    if dataset.is_regression:
        dataset.data["Y"] = {k: v.float() for k, v in dataset.Y.items()}
    Y_train = dataset.Y["train"].to(
        torch.float if dataset.is_regression else torch.long
    )

    model = ModernNCA(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **C.model,
    )
    report["n_parameters"] = lib.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    # ModernNCA returns log-probabilities internally for NLL training. They are
    # converted to probabilities before metrics and persistence, so ensembles
    # can average them directly without applying sigmoid/softmax again.
    report["prediction_type"] = None if dataset.is_regression else "probs"
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = lib.make_optimizer(model, **C.optimizer)
    train_size = dataset.size("train")
    train_indices = torch.arange(train_size, device=device)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = SummaryWriter(output)

    def get_Xy(part: str, idx) -> tuple[dict[str, Tensor], Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith("X_")
            },
            dataset.Y[part],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )

    def apply_model(part: str, idx: Tensor) -> Tensor:
        x, y = get_Xy(part, idx)
        candidate_indices = train_indices
        is_train = part == "train"
        if is_train:
            # Remove query rows before forward; forward prepends them and then
            # masks its diagonal to implement leakage-free leave-one-out NCA.
            candidate_indices = candidate_indices[
                ~torch.isin(candidate_indices, idx)
            ]
        candidate_x, candidate_y = get_Xy(
            "train", None if candidate_indices is train_indices else candidate_indices
        )
        return model(
            x_=x,
            y=y if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            is_train=is_train,
        )

    def loss_fn(prediction: Tensor, target: Tensor) -> Tensor:
        return (
            F.mse_loss(prediction, target)
            if dataset.is_regression
            else F.nll_loss(prediction, target.long())
        )

    @torch.inference_mode()
    def evaluate(parts: list[str], current_eval_batch_size: int):
        model.eval()
        predictions = {}
        for part in parts:
            while current_eval_batch_size:
                try:
                    raw_prediction = torch.cat(
                        [
                            apply_model(part, idx)
                            for idx in torch.arange(
                                dataset.size(part), device=device
                            ).split(current_eval_batch_size)
                        ]
                    )
                    prediction = (
                        raw_prediction
                        if dataset.is_regression
                        else raw_prediction.exp()
                    )
                    if dataset.is_binclass:
                        prediction = prediction[:, 1]
                    predictions[part] = prediction.cpu().numpy()
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    current_eval_batch_size //= 2
                    logger.warning(
                        f"eval_batch_size = {current_eval_batch_size}"
                    )
                else:
                    break
            if not current_eval_batch_size:
                raise RuntimeError("Not enough memory even for eval_batch_size=1")

        metrics = (
            dataset.calculate_metrics(predictions, report["prediction_type"])
            if lib.are_valid_predictions(predictions)
            else {part: {"score": -999999.0} for part in predictions}
        )
        return metrics, predictions, current_eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "random_state": delu.random.get_state(),
                "progress": progress,
                "report": report,
                "timer": timer,
                "training_log": training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f"[...] {lib.try_get_relative_path(output)} | {timer}")
        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f"Epoch {epoch}",
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: loss_fn(apply_model("train", idx), Y_train[idx]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f"chunk_size = {chunk_size}")

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ["val", "test"], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {"epoch-losses": epoch_losses, "metrics": metrics, "time": timer()}
        )
        writer.add_scalars("loss", {"train": mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars(
                "score", {part: metrics[part]["score"]}, epoch, timer()
            )

        progress.update(metrics["val"]["score"])
        if progress.success:
            lib.celebrate()
            report["best_epoch"] = epoch
            report["metrics"] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)
        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1

    report["time"] = str(timer)
    model.load_state_dict(lib.load_checkpoint(output)["model"])
    report["metrics"], predictions, _ = evaluate(
        ["train", "val", "test"], eval_batch_size
    )
    report["chunk_size"] = chunk_size
    report["eval_batch_size"] = eval_batch_size
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    writer.close()
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.configure_libraries()
    lib.run_Function_cli(main)
