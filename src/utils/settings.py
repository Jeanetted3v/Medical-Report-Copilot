from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv(), override=True)


class Settings(BaseSettings):
    """Settings for the FastAPI application."""

    model_config = {
        "env_file": find_dotenv(),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }
    CONFIG_DIR: str = "../../config"
    LLM_PROVIDER: str
    LLM_MODEL: str
    OPENAI_API_KEY: str

    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_API_VERSION: str
    VERTEXAI_PROJECT: str
    VERTEXAI_LOCATION: str
    GOOGLE_APPLICATION_CREDENTIALS: str

    LLM_TEMPERATURE: float

    TRACELOOP_API_KEY: str


SETTINGS = Settings()