from typing import Dict, Any, List, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CompetitiveLanguageCreation:
    def __init__(self, llm_providers: List[Callable], llm_names: List[str]):
        if len(llm_providers) != len(llm_names):
            raise ValueError("The number of LLM providers must match the number of LLM names")
        self.llm_providers = llm_providers
        self.llm_names = llm_names
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer()

    def compete(self, rounds: int = 5, max_tokens: int = 100) -> Dict[str, Any]:
        self.logger.info(f"Starting competition with {len(self.llm_providers)} LLMs for {rounds} rounds")
        languages = self._create_languages()
        conversation = []
        current_context = "Multiple beings from different planets meet for the first time and try to communicate."

        for round_num in range(rounds):
            self.logger.debug(f"Starting round {round_num + 1}")
            round_responses = self._conduct_round(languages, current_context, conversation, max_tokens)
            conversation.extend(round_responses)
            current_context = self._update_context(current_context, round_responses[-1])

        analysis = self._analyze_competitive_conversation(conversation, languages)
        
        self.logger.info("Competition completed")
        return {
            "languages": languages,
            "conversation": conversation,
            "analysis": analysis
        }

    def _create_languages(self) -> List[Dict[str, str]]:
        self.logger.info("Creating languages for each LLM")
        prompt = ("Create a unique and abstract language with its own grammar and vocabulary. "
                  "Provide a brief description of the language and 5-10 key rules or features. "
                  "The language should be designed for abstract communication about complex topics.")
        
        with ThreadPoolExecutor() as executor:
            future_to_llm = {executor.submit(self._create_single_language, llm, prompt): i 
                             for i, llm in enumerate(self.llm_providers)}
            languages = [None] * len(self.llm_providers)
            for future in as_completed(future_to_llm):
                llm_index = future_to_llm[future]
                try:
                    languages[llm_index] = future.result()
                except Exception as exc:
                    self.logger.error(f"LLM {self.llm_names[llm_index]} generated an exception: {exc}")
                    languages[llm_index] = {"description": "Error", "rules": "Error occurred during language creation"}
        
        return languages

    def _create_single_language(self, llm: Callable, prompt: str) -> Dict[str, str]:
        response = llm(prompt, max_tokens=300)
        lines = response.strip().split("\n")
        return {
            "description": lines[0] if lines else "No description provided",
            "rules": "\n".join(lines[1:]) if len(lines) > 1 else "No rules provided"
        }

    def _conduct_round(self, languages: List[Dict[str, str]], current_context: str, 
                       conversation: List[str], max_tokens: int) -> List[str]:
        round_responses = []
        for i, llm in enumerate(self.llm_providers):
            prompt = self._create_prompt(languages[i]['rules'], current_context, conversation)
            try:
                response = llm(prompt, max_tokens=max_tokens)
                round_responses.append(response)
            except Exception as e:
                self.logger.error(f"Error getting response from {self.llm_names[i]}: {e}")
                round_responses.append(f"Error: {self.llm_names[i]} failed to respond")
        return round_responses

    def _create_prompt(self, language_rules: str, current_context: str, conversation: List[str]) -> str:
        prompt = f"Using the language rules:\n{language_rules}\n\nCurrent context: {current_context}\n\nRespond using your created language:"
        if conversation:
            prompt += f"\n\nThe last being said: '{conversation[-1]}'"
        return prompt

    def _update_context(self, current_context: str, response: str) -> str:
        update_prompt = f"Given the current context: '{current_context}'\n\nAnd the latest communication: '{response}'\n\nProvide a brief update to the context that captures the essence of this interaction:"
        try:
            return self.llm_providers[0](update_prompt, max_tokens=100)
        except Exception as e:
            self.logger.error(f"Error updating context: {e}")
            return current_context

    def _analyze_competitive_conversation(self, conversation: List[str], languages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            consistency_scores = self._calculate_consistency_scores(conversation)
            complexity_scores = self._calculate_complexity_scores(conversation)
            adherence_scores = self._calculate_language_adherence(conversation, languages)

            return {
                "consistency_scores": consistency_scores,
                "complexity_scores": complexity_scores,
                "adherence_scores": adherence_scores,
                "winner": self._determine_winner(consistency_scores, complexity_scores, adherence_scores)
            }
        except Exception as e:
            self.logger.error(f"Error in conversation analysis: {e}")
            return {"error": str(e)}

    def _calculate_consistency_scores(self, conversation: List[str]) -> List[float]:
        if len(conversation) < 2:
            return [0.0] * len(conversation)
        
        tfidf_matrix = self.vectorizer.fit_transform(conversation)
        cosine_similarities = cosine_similarity(tfidf_matrix)
        
        consistency_scores = []
        for i in range(1, len(conversation)):
            consistency_scores.append(cosine_similarities[i-1][i])
        
        # Add a zero for the first utterance to maintain list length
        consistency_scores.insert(0, 0.0)
        return consistency_scores

    def _calculate_complexity_scores(self, conversation: List[str]) -> List[float]:
        return [len(set(utterance.split())) / max(len(utterance.split()), 1) for utterance in conversation]

    def _calculate_language_adherence(self, conversation: List[str], languages: List[Dict[str, str]]) -> List[float]:
        adherence_scores = []
        for i, utterance in enumerate(conversation):
            language = languages[i % len(languages)]
            adherence_score = self._check_adherence(utterance, language['rules'])
            adherence_scores.append(adherence_score)
        return adherence_scores

    def _check_adherence(self, utterance: str, rules: str) -> float:
        rule_words = set(rules.lower().split())
        utterance_words = set(utterance.lower().split())
        return len(rule_words.intersection(utterance_words)) / max(len(rule_words), 1)

    def _determine_winner(self, consistency_scores, complexity_scores, adherence_scores):
        num_llms = len(self.llm_providers)
        total_scores = [0] * num_llms
        
        for i in range(num_llms):
            total_scores[i] = (
                np.mean(consistency_scores[i::num_llms]) +
                np.mean(complexity_scores[i::num_llms]) +
                np.mean(adherence_scores[i::num_llms])
            )
        
        winner_index = np.argmax(total_scores)
        return self.llm_names[winner_index]