"""Generate the complete three-image social benchmark set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .benchmark_receipt import (
    create_benchmark_receipt,
)
from .social_card import (
    create_social_card,
)
from .speedup_graph import (
    create_speedup_graph,
)


@dataclass(frozen=True, slots=True)
class SocialAssetBundle:
    """Paths to the three generated social benchmark images."""

    social_card: Path
    speedup_graph: Path
    benchmark_receipt: Path

    @property
    def image_files(
        self,
    ) -> tuple[
        Path,
        Path,
        Path,
    ]:
        """Return images in their intended carousel order."""

        return (
            self.social_card,
            self.speedup_graph,
            self.benchmark_receipt,
        )


def create_social_assets(
    run_directory: str | Path,
) -> SocialAssetBundle:
    """Regenerate all three images from one saved benchmark run."""

    directory = Path(
        run_directory
    ).expanduser().resolve()

    social_card = create_social_card(
        directory
    )
    speedup_graph = create_speedup_graph(
        directory
    )
    benchmark_receipt = (
        create_benchmark_receipt(
            directory
        )
    )

    return SocialAssetBundle(
        social_card=(
            social_card.image_file
        ),
        speedup_graph=(
            speedup_graph.image_file
        ),
        benchmark_receipt=(
            benchmark_receipt.image_file
        ),
    )
