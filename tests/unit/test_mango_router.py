import pytest
import random
from planning.mango_router import MangoRouter

def test_mango_router_initialization_and_priors():
    """Verify that adding candidate URLs correctly bootstraps Beta priors."""
    router = MangoRouter(prior_strength=4.0)
    
    candidates = {
        "https://example.com/high": 0.8,
        "https://example.com/low": 0.2
    }
    
    router.add_candidates(candidates)
    
    assert "https://example.com/high" in router.arms
    assert "https://example.com/low" in router.arms
    
    high_arm = router.arms["https://example.com/high"]
    low_arm = router.arms["https://example.com/low"]
    
    # Prior bootstrap formulas:
    # alpha = 1.0 + score * prior_strength
    # beta = 1.0 + (1.0 - score) * prior_strength
    # For high (score 0.8): alpha = 1.0 + 3.2 = 4.2, beta = 1.0 + 0.8 = 1.8
    # For low (score 0.2): alpha = 1.0 + 0.8 = 1.8, beta = 1.0 + 3.2 = 4.2
    assert pytest.approx(high_arm["alpha"]) == 4.2
    assert pytest.approx(high_arm["beta"]) == 1.8
    assert pytest.approx(low_arm["alpha"]) == 1.8
    assert pytest.approx(low_arm["beta"]) == 4.2
    
    assert high_arm["scraped"] is False
    assert low_arm["scraped"] is False

def test_mango_router_selection_removes_scraped():
    """Ensure that selected and scraped URLs are not selected again."""
    router = MangoRouter()
    candidates = {
        "https://url1.com": 0.5,
        "https://url2.com": 0.5,
        "https://url3.com": 0.5
    }
    router.add_candidates(candidates)
    
    selected_urls = []
    for _ in range(3):
        url = router.select_url()
        assert url is not None
        assert url not in selected_urls
        selected_urls.append(url)
        router.update_reward(url, 0.5)
        
    # All are now scraped, next selection should be None
    assert router.select_url() is None

def test_mango_router_reward_updates():
    """Verify that updating reward shifts alpha and beta parameters correctly."""
    router = MangoRouter()
    router.add_candidates({"https://example.com": 0.5})
    
    initial_alpha = router.arms["https://example.com"]["alpha"]
    initial_beta = router.arms["https://example.com"]["beta"]
    
    # Reward of 0.9 (highly relevant/factual)
    router.update_reward("https://example.com", 0.9)
    
    arm = router.arms["https://example.com"]
    assert arm["scraped"] is True
    assert arm["rewards"] == [0.9]
    assert pytest.approx(arm["alpha"]) == initial_alpha + 0.9
    assert pytest.approx(arm["beta"]) == initial_beta + 0.1

def test_mango_router_summary():
    """Verify that get_summary returns arms sorted by expected value."""
    router = MangoRouter()
    candidates = {
        "https://low.com": 0.1,
        "https://high.com": 0.9,
    }
    router.add_candidates(candidates)
    
    summary = router.get_summary()
    assert len(summary) == 2
    assert summary[0]["url"] == "https://high.com"
    assert summary[0]["expected_value"] > summary[1]["expected_value"]
    assert summary[0]["rewards"] == []

def test_mango_router_empty_or_duplicate():
    """Test edge cases with empty candidates or duplicate additions."""
    router = MangoRouter()
    assert router.select_url() is None
    
    router.add_candidates({"https://dup.com": 0.5})
    router.add_candidates({"https://dup.com": 0.9}) # Duplicate, should be ignored
    
    assert router.arms["https://dup.com"]["prior_score"] == 0.5
