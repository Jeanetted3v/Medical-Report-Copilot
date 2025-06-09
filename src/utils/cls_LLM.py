import json
from typing import Type, TypeVar
import litellm
import instructor
from pydantic import BaseModel
from traceloop.sdk import Traceloop
from src.utils.settings import SETTINGS


T = TypeVar('T', bound=BaseModel)

Traceloop.init(disable_batch=True)


class LLMClient:
    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        messages: list = [],
        extra_params=None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.messages = messages
        self.extra_params = extra_params
        self.instructor_client = instructor.from_litellm(litellm.acompletion)

    async def async_structured_completion(
        self, response_model: Type[T], temperature: float = 0.3
    ) -> T:
        """Async method for structured responses using instructor"""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "response_model": response_model,
            "temperature": temperature
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.extra_params:
            kwargs.update(self.extra_params)
        
        return await self.instructor_client.chat.completions.create(**kwargs)


def build_llm_client(settings_dict: dict, prompt: str) -> LLMClient:
    provider = settings_dict["provider"].lower()

    if provider == "azure":
        return LLMClient(
            provider=provider,
            api_key=settings_dict["azure_openai_api_key"],
            model=settings_dict["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            extra_params={
                "api_base": settings_dict["azure_openai_endpoint"],
                "deployment_id": settings_dict["azure_openai_deployment"]
            }
        )
    elif provider == "vertexai":
        vertex_credentials = None
        if settings_dict.get("google_application_credentials"):
            try:
                with open(settings_dict["google_application_credentials"], 'r') as file:
                    vertex_creds_dict = json.load(file)
                vertex_credentials = json.dumps(vertex_creds_dict)
            except Exception as e:
                print(f"Warning: Could not load Vertex AI credentials: {e}")
        
        # Format model name with vertex_ai/ prefix if not already present
        model_name = settings_dict["llm_model"]
        if not model_name.startswith("vertex_ai/"):
            model_name = f"vertex_ai/{model_name}"
        
        extra_params = {
            "vertex_project": settings_dict["vertexai_project"],
            "vertex_location": settings_dict["vertexai_location"]
        }
        
        if vertex_credentials:
            extra_params["vertex_credentials"] = vertex_credentials
        
        return LLMClient(
            provider="vertexai",
            api_key=None,  # Vertex AI uses service account credentials
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            extra_params=extra_params
        )
    else:  # assume "openai"
        return LLMClient(
            provider="openai",
            model="gpt-4o-mini",
            api_key=settings_dict["openai_api_key"],
            messages=[{"role": "user", "content": prompt}]
        )
    

def build_settings_dict() -> dict:
    """
    Build and return the LLM settings dictionary from global SETTINGS.

    Returns:
        dict: Dictionary containing LLM-related configuration values.
    """
    return {
        "provider": SETTINGS.LLM_PROVIDER,
        "openai_api_key": SETTINGS.OPENAI_API_KEY,
        "config_dir": SETTINGS.CONFIG_DIR,
        "azure_openai_api_key": SETTINGS.AZURE_OPENAI_API_KEY,
        "azure_openai_endpoint": SETTINGS.AZURE_OPENAI_ENDPOINT,
        "azure_openai_deployment": SETTINGS.AZURE_OPENAI_DEPLOYMENT,
        "azure_api_version": SETTINGS.AZURE_API_VERSION,
        "vertexai_project": SETTINGS.VERTEXAI_PROJECT,
        "vertexai_location": SETTINGS.VERTEXAI_LOCATION,
        "google_application_credentials": SETTINGS.GOOGLE_APPLICATION_CREDENTIALS,
        "llm_temperature": SETTINGS.LLM_TEMPERATURE,
        "llm_model": SETTINGS.LLM_MODEL
    }
