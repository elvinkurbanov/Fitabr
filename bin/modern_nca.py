"""ModernNCA baseline for the FiTabR experiment framework.

This is an independent soft-neighborhood model. It does not use FiTabR's
feature-attention module or TabR's retrieval/context aggregation blocks.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
    patience: Optional[int]
    n_epochs: Union[int, float]


class Model(nn.Module):
    """Deep embedding model trained with a soft nearest-neighbor objective."""

    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        num_embeddings: Optional[dict],
        d_embedding: int,
        d_block: int,
        n_blocks: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.n_classes = 2 if n_classes is None else n_classes
        self.temperature = temperature
        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )
        d_in = (
            n_num_features * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        if d_in == 0:
            raise ValueError('ModernNCA requires at least one input feature')
        self.input = nn.Linear(d_in, d_embedding)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_embedding),
                    nn.Linear(d_embedding, d_block),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_block, d_embedding),
                    nn.Dropout(dropout),
                )
                for _ in range(n_blocks)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(d_embedding), nn.Linear(d_embedding, d_embedding))

    def encode(self, x_: dict[str, Tensor]) -> Tensor:
        features = []
        x_num = x_.get('num')
        if x_num is not None:
            features.append(
                x_num if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        x_bin = x_.get('bin')
        if x_bin is not None:
            features.append(x_bin)
        x_cat = x_.get('cat')
        if x_cat is not None:
            assert self.one_hot_encoder is not None
            features.append(self.one_hot_encoder(x_cat))
        h = self.input(torch.cat(features, dim=1))
        for block in self.blocks:
            h = h + block(h)
        return F.normalize(self.output(h), p=2, dim=1)

    def forward(
        self,
        *,
        x_: dict[str, Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        query_ids: Optional[Tensor] = None,
        candidate_ids: Optional[Tensor] = None,
    ) -> Tensor:
        query = self.encode(x_)
        candidates = self.encode(candidate_x_)
        distances = torch.cdist(query, candidates, p=2).square()
        scores = -distances / self.temperature
        if query_ids is not None and candidate_ids is not None:
            scores = scores.masked_fill(
                query_ids[:, None] == candidate_ids[None, :], float('-inf')
            )
        weights = torch.softmax(scores, dim=1)
        labels = F.one_hot(candidate_y.long().reshape(-1), self.n_classes).to(weights.dtype)
        probabilities = (weights @ labels).clamp_min(1e-7)
        if self.n_classes == 2:
            p_positive = probabilities[:, 1].clamp(1e-7, 1 - 1e-7)
            return torch.logit(p_positive)
        return probabilities.log()


@torch.inference_mode()
def _predict(model, dataset, part: str, candidate_chunk_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    candidates = {
        key[2:]: value['train'] for key, value in dataset.data.items()
        if key.startswith('X_')
    }
    candidate_y = dataset.Y['train'].reshape(-1).long()
    outputs = []
    x_part = {
        key[2:]: value[part] for key, value in dataset.data.items()
        if key.startswith('X_')
    }
    for start in range(0, dataset.size(part), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, dataset.size(part))
        x = {key: value[start:stop] for key, value in x_part.items()}
        outputs.append(model(x_=x, candidate_x_=candidates, candidate_y=candidate_y).cpu())
    return torch.cat(outputs).numpy()


def _make_candidate_indices(
    query_ids: Tensor, y_train_cpu: Tensor, sample_rate: float, device: torch.device
) -> Tensor:
    """Sample a shared neighborhood and ensure every query has a positive neighbor."""
    n_train = len(y_train_cpu)
    n_sample = min(n_train, max(2, int(np.ceil(n_train * sample_rate))))
    selected = torch.randperm(n_train)[:n_sample].tolist()
    selected_set = set(selected)
    queries = query_ids.detach().cpu().tolist()
    for query_id in queries:
        positive = torch.nonzero(y_train_cpu == y_train_cpu[query_id]).reshape(-1).tolist()
        positive = [idx for idx in positive if idx != query_id]
        if positive and not any(idx in selected_set for idx in positive):
            chosen = positive[int(torch.randint(len(positive), (1,)).item())]
            selected_set.add(chosen)
    return torch.tensor(sorted(selected_set), dtype=torch.long, device=device)


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
    if dataset.task_type not in ('binclass', 'multiclass'):
        raise ValueError('This ModernNCA adaptation currently supports classification only')
    y_train = dataset.Y['train'].reshape(-1).long()
    y_train_cpu = y_train.detach().cpu()
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **C.model,
    ).to(device)
    report['n_parameters'] = lib.get_n_parameters(model)
    report['prediction_type'] = 'logits'
    optimizer = lib.make_optimizer(model, **C.optimizer)
    train_size = dataset.size('train')
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)
    epoch = 0
    timer = lib.run_timer()

    def get_features(part: str, indices: Optional[Tensor] = None) -> dict[str, Tensor]:
        values = {
            key[2:]: value[part] for key, value in dataset.data.items()
            if key.startswith('X_')
        }
        return values if indices is None else {key: value[indices] for key, value in values.items()}

    def evaluate(parts: list[str]):
        model.eval()
        predictions = {part: _predict(model, dataset, part) for part in parts}
        metrics = dataset.calculate_metrics(predictions, report['prediction_type'])
        return metrics, predictions

    while epoch < C.n_epochs:
        model.train()
        losses = []
        for query_ids in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            candidate_ids = _make_candidate_indices(
                query_ids, y_train_cpu, C.model['sample_rate'], device
            )
            query_x = get_features('train', query_ids)
            candidate_x = get_features('train', candidate_ids)
            logits = model(
                x_=query_x,
                candidate_x_=candidate_x,
                candidate_y=y_train[candidate_ids],
                query_ids=query_ids,
                candidate_ids=candidate_ids,
            )
            y = y_train[query_ids]
            if dataset.is_multiclass:
                loss = F.nll_loss(logits, y)
            else:
                loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), y.float().reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())

        mean_loss = torch.stack(losses).mean().item()
        metrics, predictions = evaluate(['val', 'test'])
        lib.print_metrics(mean_loss, metrics)
        training_log.append({'metrics': metrics, 'time': timer()})
        writer.add_scalars('loss', {'train': mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars('score', {part: metrics[part]['score']}, epoch, timer())
        progress.update(metrics['val']['score'])
        if progress.success:
            report['best_epoch'] = epoch
            report['metrics'] = metrics
            lib.dump_checkpoint(
                {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'progress': progress, 'report': report, 'training_log': training_log}, output
            )
            lib.dump_predictions(predictions, output)
            lib.dump_report(report, output)
            lib.backup_output(output)
        elif progress.fail or not lib.are_valid_predictions(predictions):
            break
        epoch += 1
    report['time'] = str(timer)
    checkpoint = lib.load_checkpoint(output)
    model.load_state_dict(checkpoint['model'])
    report['metrics'], predictions = evaluate(['train', 'val', 'test'])
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    lib.dump_report(report, output)
    lib.finish(output, report)
    writer.close()
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
