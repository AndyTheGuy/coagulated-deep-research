import random
import structlog
from typing import Any, Dict, List, Optional

logger = structlog.get_logger("deep-research")

class MangoRouter:
    """Thompson Sampling Multi-Armed Bandit starting-point optimizer (Mango).
    Models candidate starting-point URLs as Bandit arms with Beta distributions to maximize
    relevance and factuality density while avoiding redundant web traversal.
    """
    
    def __init__(self, prior_strength: float = 4.0):
        self.prior_strength = prior_strength
        # Maps url -> {"alpha": float, "beta": float, "scraped": bool, "prior_score": float}
        self.arms: Dict[str, Dict[str, Any]] = {}

    def add_candidates(self, urls_with_heuristics: Dict[str, float]):
        """Adds a list of candidate starting URLs with initial heuristic scores.
        
        Args:
            urls_with_heuristics: Dictionary of url -> heuristic_score (in [0.0, 1.0])
        """
        for url, score in urls_with_heuristics.items():
            if url in self.arms:
                continue
                
            # Bound score to prevent divide-by-zero or extreme priors
            score = max(0.05, min(0.95, score))
            
            # Bootstrap priors using heuristic score and strength
            alpha = 1.0 + score * self.prior_strength
            beta = 1.0 + (1.0 - score) * self.prior_strength
            
            self.arms[url] = {
                "alpha": alpha,
                "beta": beta,
                "scraped": False,
                "prior_score": score,
                "rewards": []
            }
            logger.debug("Added Mango bandit arm", url=url, alpha=alpha, beta=beta, score=score)

    def select_url(self) -> Optional[str]:
        """Chooses the next URL to scrape using Thompson Sampling from Beta distributions."""
        best_url = None
        best_sample = -1.0
        
        unscraped_arms = {url: arm for url, arm in self.arms.items() if not arm["scraped"]}
        
        if not unscraped_arms:
            logger.debug("No unscraped Mango bandit arms remaining.")
            return None
            
            
        for url, arm in unscraped_arms.items():
            # Draw sample from Beta(alpha, beta) using random.betavariate
            try:
                sample = random.betavariate(arm["alpha"], arm["beta"])
            except ValueError:
                # Fallback if params became invalid somehow
                sample = random.uniform(0.0, 1.0)
                
            if sample > best_sample:
                best_sample = sample
                best_url = url
                
        logger.info("Mango Thompson Sampling selected URL", url=best_url, sample=best_sample)
        return best_url

    def update_reward(self, url: str, reward: float):
        """Updates the posterior Beta distribution of a selected URL based on scraping feedback.
        
        Args:
            url: The selected URL that was scraped.
            reward: Factual density or relevance reward (in [0.0, 1.0]).
        """
        if url not in self.arms:
            logger.warning("Attempted to update reward for untracked URL in Mango", url=url)
            return
            
        reward = max(0.0, min(1.0, reward))
        arm = self.arms[url]
        
        # Update Beta distribution parameters
        arm["alpha"] += reward
        arm["beta"] += (1.0 - reward)
        arm["scraped"] = True
        arm["rewards"].append(reward)
        
        logger.info("Mango Thompson Sampling updated reward", url=url, reward=reward, posterior_alpha=arm["alpha"], posterior_beta=arm["beta"])

    def get_summary(self) -> List[Dict[str, Any]]:
        """Returns a summarized list of all tracked arms and their performance."""
        summary = []
        for url, arm in self.arms.items():
            expected_value = arm["alpha"] / (arm["alpha"] + arm["beta"])
            summary.append({
                "url": url,
                "scraped": arm["scraped"],
                "prior_score": arm["prior_score"],
                "expected_value": expected_value,
                "rewards": arm["rewards"]
            })
        return sorted(summary, key=lambda x: x["expected_value"], reverse=True)
