"""To run:
poetry run python -m src.main
"""
import logging
import hydra
import asyncio
from omegaconf import DictConfig
from src.utils.settings import SETTINGS
from src.utils.logging import setup_logging
from utils.cls_LLM import build_settings_dict
from src.pipeline import MainPipeline
from pathlib import Path

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()


async def quick_check_pipeline(cfg):
    """Quick check pipeline for testing purposes."""
    settings_dict = build_settings_dict()
    pipeline = MainPipeline(cfg, settings_dict)
    
    pdf_path = "./data/raw/Eye_test_report (1).pdf"
    if not Path(pdf_path).exists():
        logger.error(f"PDF file {pdf_path} does not exist.")
        return
    
    pages = pipeline.pdf_to_images(pdf_path)
    is_medical_image = await pipeline.check_medical_images(pages)
    if is_medical_image:
        interpretation = await pipeline.generate_image_interpretation(pages)
    logger.info(f"Medical image report interpretation: {interpretation}")


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


