import yaml
import json
from typing import List, Dict, Any
import logging
from llm_providers import get_llm_provider
from latent_explorer import LatentExplorer
from competitive_language import CompetitiveLanguageCreation

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_latent_exploration(config: Dict[str, Any], llm_provider: Any) -> Dict[str, Any]:
    explorer = LatentExplorer(config, llm_provider)
    results = explorer.explore(num_iterations=config.get('exploration_iterations', 5))
    summary = explorer.summarize_exploration(results)
    return summary

def run_competitive_language_creation(config: Dict[str, Any], llm_providers: List[Any]) -> Dict[str, Any]:
    clc = CompetitiveLanguageCreation(llm_providers, config.get('llm_names', []))
    results = clc.compete(rounds=config.get('competition_rounds', 3), max_tokens=config.get('max_tokens', 150))
    return results

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config('config.yaml')
    
    logger.debug(f"Loaded config: {json.dumps(config, indent=2)}")
    
    llm_providers = []
    for provider_config in config.get('llm_providers', []):
        logger.debug(f"Processing provider config: {provider_config}")
        if not isinstance(provider_config, dict):
            logger.error(f"Invalid provider config: {provider_config}. Expected a dictionary.")
            continue
        try:
            provider = get_llm_provider(provider_config)
            llm_providers.append(provider)
        except Exception as e:
            logger.error(f"Failed to initialize provider: {provider_config.get('type', 'Unknown')}. Error: {str(e)}")
    
    if not llm_providers:
        logger.error("No LLM providers were successfully initialized. Exiting.")
        return

    logger.info("Starting Latent Exploration")
    exploration_summary = run_latent_exploration(config, llm_providers[0])  # Using the first provider for exploration
    
    logger.info("Starting Competitive Language Creation")
    competition_results = run_competitive_language_creation(config, llm_providers)
    
    # Combine results
    final_results = {
        "latent_exploration": exploration_summary,
        "competitive_language_creation": competition_results
    }
    
    # Save results to a file
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("Results saved to results.json")

if __name__ == "__main__":
    main()