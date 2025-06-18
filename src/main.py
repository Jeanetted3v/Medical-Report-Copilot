"""To run:
poetry run python -m src.main
python -m src.main
"""
import logging
import hydra
import asyncio
from omegaconf import DictConfig
from src.utils.settings import SETTINGS
from src.utils.logging import setup_logging
from src.pipeline import MainPipeline

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()


async def quick_check_pipeline(cfg: DictConfig) -> None:
    pipeline = MainPipeline(cfg)
    await pipeline.run_single_pdf(cfg.pdf_path)
    return None


@hydra.main(
    version_base=None,
    config_path=SETTINGS.CONFIG_DIR,
    config_name="config"
)
def main(cfg: DictConfig) -> None:
    """Main function to run the application with Hydra configuration.

    Args:
        cfg (DictConfig): Configuration object from Hydra.
    """
    asyncio.run(quick_check_pipeline(cfg))


if __name__ == "__main__":
    main()


