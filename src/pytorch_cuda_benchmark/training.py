# pyright: reportUnknownMemberType=false

"""Device-aware training and timing utilities."""

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Measurements collected from one timed training run."""

    device: str
    mixed_precision: bool
    epochs: int
    processed_batches: int
    processed_samples: int
    mean_loss: float
    elapsed_seconds: float
    samples_per_second: float


def train_and_measure(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str | torch.device,
    epochs: int,
    warmup_batches: int,
    mixed_precision: bool,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> TrainingResult:
    """Train a model and measure its steady-state training performance."""

    resolved_device = torch.device(device)

    _validate_arguments(
        data_loader=data_loader,
        device=resolved_device,
        epochs=epochs,
        warmup_batches=warmup_batches,
        learning_rate=learning_rate,
        momentum=momentum,
    )

    use_mixed_precision = (
        mixed_precision and resolved_device.type == "cuda"
    )

    model.to(resolved_device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    scaler: torch.amp.GradScaler | None = None

    if use_mixed_precision:
        scaler = torch.amp.GradScaler("cuda")

    _run_warmup(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=resolved_device,
        warmup_batches=warmup_batches,
        use_mixed_precision=use_mixed_precision,
    )

    _synchronize(resolved_device)
    start_time = perf_counter()

    total_loss = 0.0
    processed_batches = 0
    processed_samples = 0

    for _ in range(epochs):
        for images, targets in data_loader:
            batch_loss, batch_size = _train_batch(
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=resolved_device,
                use_mixed_precision=use_mixed_precision,
            )

            total_loss += batch_loss * batch_size
            processed_batches += 1
            processed_samples += batch_size

    _synchronize(resolved_device)
    elapsed_seconds = perf_counter() - start_time

    mean_loss = total_loss / processed_samples
    samples_per_second = processed_samples / elapsed_seconds

    return TrainingResult(
        device=str(resolved_device),
        mixed_precision=use_mixed_precision,
        epochs=epochs,
        processed_batches=processed_batches,
        processed_samples=processed_samples,
        mean_loss=mean_loss,
        elapsed_seconds=elapsed_seconds,
        samples_per_second=samples_per_second,
    )


def _run_warmup(
    *,
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.CrossEntropyLoss,
    optimizer: SGD,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    warmup_batches: int,
    use_mixed_precision: bool,
) -> None:
    """Execute untimed batches before collecting measurements."""

    for batch_index, (images, targets) in enumerate(data_loader):
        if batch_index >= warmup_batches:
            break

        _train_batch(
            model=model,
            images=images,
            targets=targets,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )


def _train_batch(
    *,
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    optimizer: SGD,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    use_mixed_precision: bool,
) -> tuple[float, int]:
    """Execute one forward, backward, and optimizer step."""

    non_blocking = device.type == "cuda"

    images = images.to(device, non_blocking=non_blocking)
    targets = targets.to(device, non_blocking=non_blocking)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        enabled=use_mixed_precision,
    ):
        predictions = model(images)
        loss = criterion(predictions, targets)

    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return float(loss.detach().item()), targets.size(0)


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work before reading the wall-clock timer."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_arguments(
    *,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
    warmup_batches: int,
    learning_rate: float,
    momentum: float,
) -> None:
    """Validate training and timing arguments."""

    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Device must be either 'cpu' or 'cuda'.")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but CUDA is not available."
        )

    if epochs <= 0:
        raise ValueError("Epochs must be greater than zero.")

    if warmup_batches < 0:
        raise ValueError("Warm-up batches cannot be negative.")

    if len(data_loader) == 0:
        raise ValueError("The data loader must contain at least one batch.")

    if warmup_batches > len(data_loader):
        raise ValueError(
            "Warm-up batches cannot exceed the data loader length."
        )

    if learning_rate <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    if not 0 <= momentum < 1:
        raise ValueError("Momentum must be between zero and one.")
