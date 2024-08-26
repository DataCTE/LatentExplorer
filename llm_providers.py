from abc import ABC, abstractmethod
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
import logging
import torch

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_text(self, prompt: str, max_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return f"Error: Unable to get response from OpenAI API"

class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    def generate_text(self, prompt: str, max_tokens: int) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Hugging Face model generation failed: {e}")
            return f"Error: Unable to get response from Hugging Face model"

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_text(self, prompt: str, max_tokens: int) -> str:
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                max_tokens_to_sample=max_tokens,
            )
            return response.completion.strip()
        except Exception as e:
            logger.error(f"Anthropic API request failed: {e}")
            return f"Error: Unable to get response from Anthropic API"

def get_llm_provider(config):
    logger.debug(f"Initializing provider with config: {config}")
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config type. Expected dict, got {type(config)}")
    
    provider_type = config.get('type')
    if not provider_type:
        raise ValueError("Provider type not specified in config")

    if provider_type == 'openai':
        return OpenAIProvider(config['api_key'], config['model'])
    elif provider_type == 'huggingface':
        return HuggingFaceProvider(config['model_name'])
    elif provider_type == 'anthropic':
        return AnthropicProvider(config['api_key'], config['model'])
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}")