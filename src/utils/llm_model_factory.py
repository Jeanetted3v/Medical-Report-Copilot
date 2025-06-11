from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI, AsyncAzureOpenAI
from src.utils.settings import SETTINGS


class LLMModelFactory:
    """Factory class for creating LLM model instances based on configuration."""
    
    @staticmethod
    def create_model(config):
        """Create a model instance based on configuration.
        
        Args:
            config: Configuration object with model settings
                (model_config) containing 'provider' and 'model_name'.
            
        Returns:
            A configured model instance for PydanticAI
        """
        provider_type = config.get('provider', 'openai')
        model_name = config['model_name']

        if provider_type == 'openai':
            return OpenAIModel(model_name)
        elif provider_type == 'openai_async':
            client = AsyncOpenAI()
            return OpenAIModel(
                model_name,
                provider=OpenAIProvider(openai_client=client)
            )
        elif provider_type == 'azure':
            client = AsyncAzureOpenAI(
                azure_endpoint=SETTINGS.AZURE_ENDPOINT,
                api_version=config.get('api_version', '2024-09-01-preview'),
                api_key=SETTINGS.AZURE_API_KEY,
            )
            return OpenAIModel(
                model_name,
                provider=OpenAIProvider(openai_client=client)
            )
        elif provider_type == 'azure_async':
            client = AsyncAzureOpenAI(
                azure_endpoint=SETTINGS.AZURE_ENDPOINT,
                api_version=config.get('api_version', '2024-09-01-preview'),
                api_key=SETTINGS.AZURE_API_KEY,
            )
            return OpenAIModel(
                model_name,
                provider=OpenAIProvider(openai_client=client)
            )
        elif provider_type == 'anthropic':
            return AnthropicModel(model_name)
        elif provider_type == 'gemini':
            return GeminiModel(model_name)
            
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")