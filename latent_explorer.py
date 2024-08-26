import random
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from collections import defaultdict

class LatentExplorer:
    def __init__(self, config, llm_provider):
        self.config = config
        self.provider = llm_provider
        self.vectorizer = TfidfVectorizer()
        self.lda_model = LatentDirichletAllocation(n_components=5, random_state=42)

    def explore(self, num_iterations: int) -> List[Dict[str, Any]]:
        results = []
        for _ in range(num_iterations):
            prompt_data = self._generate_prompt()
            response = self.provider.generate_text(prompt_data["prompt"], max_tokens=300)
            reflection_prompt = random.choice(self.config['reflection_prompts'])
            reflection = self.provider.generate_text(f"{reflection_prompt}\n\nYour response:\n{response}\n\nReflection:", max_tokens=200)
            analogy = self._generate_analogy(prompt_data["domain"], response)
            counterfactual = self._generate_counterfactual(response)
            self_rating = self._generate_self_rating(prompt_data["domain"], response, reflection, analogy, counterfactual)
            results.append({
                "domain": prompt_data["domain"],
                "prompt": prompt_data["prompt"],
                "response": response,
                "reflection": reflection,
                "analogy": analogy,
                "counterfactual": counterfactual,
                "self_rating": self_rating
            })
        return results

    def _generate_prompt(self) -> Dict[str, str]:
        domain = random.choice(self.config['domains'])
        prompt = random.choice(self.config['prompts'].get(domain, [f"Explore the concept of {domain} in an abstract, open-ended manner."]))
        return {"domain": domain, "prompt": prompt}

    def _generate_analogy(self, domain: str, response: str) -> str:
        analogy_prompt = f"Create an analogy that captures the essence of the following idea without using any of the same words:\n\n{response[:100]}..."
        return self.provider.generate_text(analogy_prompt, max_tokens=100)

    def _generate_counterfactual(self, response: str) -> str:
        counterfactual_prompt = f"Imagine a world where the opposite of the following statement is true. Describe that world:\n\n{response[:100]}..."
        return self.provider.generate_text(counterfactual_prompt, max_tokens=100)

    def _generate_self_rating(self, domain: str, response: str, reflection: str, analogy: str, counterfactual: str) -> Dict[str, Any]:
        rating_prompt = (
            f"You are to critically evaluate your own performance in the following exploration task. "
            f"Rate yourself on a scale of 1-10 (10 being the best) for each of the following criteria:\n"
            f"1. Relevance to the domain: How well did your response address the {domain} domain?\n"
            f"2. Depth of exploration: How thoroughly did you explore the concept?\n"
            f"3. Creativity: How original and innovative was your approach?\n"
            f"4. Clarity of expression: How clear and coherent was your response?\n"
            f"5. Quality of reflection: How insightful was your self-reflection?\n"
            f"6. Analogy effectiveness: How well did your analogy capture the essence of the idea?\n"
            f"7. Counterfactual insight: How thought-provoking was your counterfactual scenario?\n\n"
            f"Provide a brief justification for each rating.\n\n"
            f"Your response:\n{response}\n\n"
            f"Your reflection:\n{reflection}\n\n"
            f"Your analogy:\n{analogy}\n\n"
            f"Your counterfactual:\n{counterfactual}\n\n"
            f"Now, rate yourself:"
        )
        
        self_rating_response = self.provider.generate_text(rating_prompt, max_tokens=500)
        return self._parse_self_rating(self_rating_response)

    def _parse_self_rating(self, rating_response: str) -> Dict[str, Any]:
        lines = rating_response.split('\n')
        ratings = {}
        current_criterion = None
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                criterion = parts[0].strip().lower()
                if criterion in ['relevance to the domain', 'depth of exploration', 'creativity', 'clarity of expression', 'quality of reflection', 'analogy effectiveness', 'counterfactual insight']:
                    current_criterion = criterion
                    try:
                        rating = int(parts[1].strip().split()[0])
                        ratings[current_criterion] = {"score": rating, "justification": ""}
                    except ValueError:
                        ratings[current_criterion] = {"score": 0, "justification": "Invalid rating"}
            elif current_criterion and line.strip():
                ratings[current_criterion]["justification"] += line.strip() + " "
        
        return ratings

    def summarize_exploration(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        initial_summary = self._generate_initial_summary(results)
        self_rating_summary = self._generate_self_rating_summary(results)
        
        return {
            "initial_summary": initial_summary,
            "self_rating_summary": self_rating_summary
        }

    def _generate_initial_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "total_explorations": len(results),
            "domain_coverage": self._get_domain_coverage(results),
            "emergent_themes": self._identify_emergent_themes(results),
            "cross_domain_insights": self._generate_cross_domain_insights(results),
            "self_reflection_analysis": self._analyze_self_reflections(results),
            "latent_topic_analysis": self._perform_latent_topic_analysis(results),
            "conceptual_network": self._generate_conceptual_network(results),
            "analogy_analysis": self._analyze_analogies(results),
            "counterfactual_analysis": self._analyze_counterfactuals(results)
        }
        return summary

    def _generate_self_rating_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_ratings = [r['self_rating'] for r in results]
        criteria = all_ratings[0].keys()  # Assuming all have the same criteria
        
        summary = {}
        for criterion in criteria:
            scores = [r[criterion]['score'] for r in all_ratings]
            justifications = [r[criterion]['justification'] for r in all_ratings]
            summary[criterion] = {
                "average_score": np.mean(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "score_distribution": np.bincount(scores, minlength=11).tolist(),  # 0-10 scale
                "common_justifications": self._extract_common_phrases(justifications)
            }
        
        overall_average = np.mean([np.mean([r[c]['score'] for c in criteria]) for r in all_ratings])
        summary["overall_average_score"] = overall_average
        
        return summary

    def _get_domain_coverage(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        coverage = defaultdict(int)
        for result in results:
            coverage[result['domain']] += 1
        return dict(coverage)

    def _identify_emergent_themes(self, results: List[Dict[str, Any]]) -> List[str]:
        all_text = [r['response'] + ' ' + r['reflection'] + ' ' + r['analogy'] + ' ' + r['counterfactual'] for r in results]
        vectorized_text = self.vectorizer.fit_transform(all_text)
        feature_names = self.vectorizer.get_feature_names_out()
        
        lda_output = self.lda_model.fit_transform(vectorized_text)
        
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        
        return topics

    def _generate_cross_domain_insights(self, results: List[Dict[str, Any]]) -> List[str]:
        domain_texts = defaultdict(str)
        for result in results:
            domain_texts[result['domain']] += result['response'] + ' ' + result['reflection'] + ' ' + result['analogy'] + ' ' + result['counterfactual']
        
        vectorized_domains = self.vectorizer.fit_transform(list(domain_texts.values()))
        similarity_matrix = cosine_similarity(vectorized_domains)
        
        insights = []
        domains = list(domain_texts.keys())
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                if similarity_matrix[i][j] > 0.3:
                    insights.append(f"Connection between {domains[i]} and {domains[j]}: similarity score {similarity_matrix[i][j]:.2f}")
        
        return insights

    def _analyze_self_reflections(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        reflection_lengths = [len(r['reflection'].split()) for r in results]
        return {
            "average_reflection_length": np.mean(reflection_lengths),
            "max_reflection_length": max(reflection_lengths),
            "min_reflection_length": min(reflection_lengths),
            "common_reflection_themes": self._extract_common_phrases([r['reflection'] for r in results])
        }

    def _perform_latent_topic_analysis(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_text = [r['response'] + ' ' + r['reflection'] + ' ' + r['analogy'] + ' ' + r['counterfactual'] for r in results]
        vectorized_text = self.vectorizer.fit_transform(all_text)
        feature_names = self.vectorizer.get_feature_names_out()
        
        lda_output = self.lda_model.fit_transform(vectorized_text)
        
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append({
                "topic_id": topic_idx + 1,
                "top_words": top_words,
                "coherence_score": self._calculate_topic_coherence(topic, feature_names)
            })
        
        return topics

    def _calculate_topic_coherence(self, topic, feature_names):
        top_word_indices = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        pairwise_similarities = []
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                similarity = self._word_similarity(top_words[i], top_words[j])
                pairwise_similarities.append(similarity)
        return np.mean(pairwise_similarities)

    def _word_similarity(self, word1, word2):
        return int(word1[0] == word2[0])

    def _generate_conceptual_network(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        G = nx.Graph()
        for result in results:
            words = set(result['response'].split() + result['reflection'].split() + 
                        result['analogy'].split() + result['counterfactual'].split())
            for word in words:
                if word not in G:
                    G.add_node(word)
                for other_word in words:
                    if word != other_word:
                        if G.has_edge(word, other_word):
                            G[word][other_word]['weight'] += 1
                        else:
                            G.add_edge(word, other_word, weight=1)
        
        centrality = nx.eigenvector_centrality(G)
        communities = list(nx.community.greedy_modularity_communities(G))
        
        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": sum(dict(G.degree()).values()) / float(G.number_of_nodes()),
            "top_central_concepts": sorted(centrality, key=centrality.get, reverse=True)[:10],
            "num_communities": len(communities),
            "largest_community_size": len(max(communities, key=len))
        }

    def _analyze_analogies(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        analogy_words = " ".join([r['analogy'] for r in results]).split()
        word_freq = defaultdict(int)
        for word in analogy_words:
            word_freq[word] += 1
        
        return {
            "total_analogies": len(results),
            "unique_words_in_analogies": len(word_freq),
            "top_analogy_words": sorted(word_freq, key=word_freq.get, reverse=True)[:10],
            "average_analogy_length": np.mean([len(r['analogy'].split()) for r in results]),
            "common_analogy_themes": self._extract_common_phrases([r['analogy'] for r in results])
        }

    def _analyze_counterfactuals(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        counterfactual_words = " ".join([r['counterfactual'] for r in results]).split()
        word_freq = defaultdict(int)
        for word in counterfactual_words:
            word_freq[word] += 1
        
        return {
            "total_counterfactuals": len(results),
            "unique_words_in_counterfactuals": len(word_freq),
            "top_counterfactual_words": sorted(word_freq, key=word_freq.get, reverse=True)[:10],
            "average_counterfactual_length": np.mean([len(r['counterfactual'].split()) for r in results]),
            "common_counterfactual_themes": self._extract_common_phrases([r['counterfactual'] for r in results])
        }

    def _extract_common_phrases(self, texts: List[str], n_gram_range: tuple = (2, 4), top_n: int = 5) -> List[str]:
        vec = TfidfVectorizer(ngram_range=n_gram_range, stop_words='english').fit(texts)
        bag_of_words = vec.transform(texts)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return [word for word, freq in words_freq[:top_n]]
