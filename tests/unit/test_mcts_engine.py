import asyncio
import math
import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessage

from planning.mcts_engine import MCTSNode, PCTSEngine

# ==========================================
# 1. MCTSNode Tests
# ==========================================

def test_mcts_node_initialization_default():
    """Test default state and parameters of MCTSNode."""
    state = {"evidence": "initial evidence", "gaps": ["gap1"]}
    node = MCTSNode(state=state, action="Investigate", depth=1)
    
    assert node.state == state
    assert node.action == "Investigate"
    assert node.parent is None
    assert node.children == []
    assert node.visits == 0
    assert node.total_reward == 0.0
    assert node.depth == 1
    assert node.is_terminal is False
    assert node.unexpanded_actions is None


def test_mcts_node_terminal_state():
    """Test various conditions under which MCTSNode is considered terminal."""
    # Terminal specified via state dict
    node_terminal_state = MCTSNode(state={"is_terminal": True}, depth=0)
    assert node_terminal_state.is_terminal is True

    # Terminal forced due to depth limit (depth >= 4)
    node_depth_limit = MCTSNode(state={}, depth=4)
    assert node_depth_limit.is_terminal is True


def test_mcts_node_uct_score():
    """Test Upper Confidence Bound Applied to Trees (UCT) calculations."""
    parent = MCTSNode(state={})
    parent.visits = 10

    child = MCTSNode(state={}, parent=parent)
    
    # 1. Unvisited node uct_score should be infinity
    assert child.uct_score() == float("inf")

    # 2. Visited node calculation with default exploration weight (1.414)
    child.visits = 2
    child.total_reward = 1.5
    
    # exploitation = 1.5 / 2 = 0.75
    # exploration = 1.414 * sqrt(log(10) / 2)
    expected_exploitation = 0.75
    expected_exploration = 1.414 * math.sqrt(math.log(10) / 2)
    expected_uct = expected_exploitation + expected_exploration
    
    assert math.isclose(child.uct_score(), expected_uct, rel_tol=1e-5)

    # 3. Custom exploration weight
    custom_weight = 2.0
    expected_custom_uct = expected_exploitation + custom_weight * math.sqrt(math.log(10) / 2)
    assert math.isclose(child.uct_score(exploration_weight=custom_weight), expected_custom_uct, rel_tol=1e-5)

    # 4. If parent is None, parent_visits falls back to self.visits
    root_node = MCTSNode(state={})
    root_node.visits = 5
    root_node.total_reward = 2.5
    
    # exploitation = 2.5 / 5 = 0.5
    # exploration = 1.414 * sqrt(log(5) / 5)
    expected_root_uct = 0.5 + 1.414 * math.sqrt(math.log(5) / 5)
    assert math.isclose(root_node.uct_score(), expected_root_uct, rel_tol=1e-5)


def test_mcts_node_best_child():
    """Test best child selection based on highest UCT score."""
    parent = MCTSNode(state={})
    parent.visits = 5

    # If children is empty, best_child returns itself
    assert parent.best_child() == parent

    child1 = MCTSNode(state={}, parent=parent)
    child1.visits = 2
    child1.total_reward = 1.0  # UCT = 0.5 + 1.414 * sqrt(log(5)/2) =~ 1.768
    
    child2 = MCTSNode(state={}, parent=parent)
    child2.visits = 3
    child2.total_reward = 2.5  # UCT = 0.8333 + 1.414 * sqrt(log(5)/3) =~ 1.869

    parent.children = [child1, child2]

    # child2 has higher UCT score and should be selected
    assert parent.best_child() == child2

    # If child1 gets updated with a massive reward, it should become the best child
    child1.total_reward = 6.0  # exploitation = 3.0, UCT = 3.0 + 1.414 * sqrt(log(5)/2) =~ 4.268
    assert parent.best_child() == child1


def test_mcts_node_update():
    """Test parent/child updates on reward backpropagation."""
    node = MCTSNode(state={})
    assert node.visits == 0
    assert node.total_reward == 0.0

    node.update(0.85)
    assert node.visits == 1
    assert node.total_reward == 0.85

    node.update(0.15)
    assert node.visits == 2
    assert node.total_reward == 1.00


# ==========================================
# 2. PCTSEngine Mock and Tests
# ==========================================

class MockLLMRouter:
    """Mock LLMRouter mimicking deep-research router interface."""
    
    def __init__(self, fail_on=None):
        self.fail_on = fail_on or []  # List of node_names to fail
        self.calls = []

    async def ainvoke(self, messages, tier="STANDARD", agent_name="MCTSPlanner", node_name=None):
        self.calls.append({
            "messages": messages,
            "tier": tier,
            "agent_name": agent_name,
            "node_name": node_name
        })

        if node_name in self.fail_on:
            raise RuntimeError(f"Mock failure in {node_name}")

        if node_name == "generate_candidate_intents":
            return AIMessage(content="Analyze competitors\nReview patents\nEvaluate pricing")
        
        elif node_name == "simulate_outcome_state":
            # Simulate a terminal condition when action contains 'Evaluate pricing'
            action_str = ""
            for msg in messages:
                if hasattr(msg, "content") and "Proposed Research Step:" in str(msg.content):
                    action_str = str(msg.content)
            
            if "Evaluate pricing" in action_str:
                return AIMessage(content="The pricing state is satisfied and fully answered.")
            else:
                return AIMessage(content="This uncovers the core product details and price points.")
        
        elif node_name == "evaluate_state_quality":
            return AIMessage(content="0.85")
        
        elif node_name == "generate_queries_for_intent":
            return AIMessage(content="competitor pricing table\ncompetitor tier structure\ncompetitor discounts")
        
        elif node_name == "repair_plan":
            return AIMessage(content="repair query 1\nrepair query 2\nrepair query 3")
        
        return AIMessage(content="default content")


@pytest.mark.asyncio
async def test_pcts_engine_search_success():
    """Test successful search execution across Selection, Expansion, Simulation and Backpropagation."""
    mock_router = MockLLMRouter()
    engine = PCTSEngine(router=mock_router)

    best_action, queries = await engine.search(
        sub_question="What is the pricing strategy of competitors?",
        topic="Competitor Analysis",
        initial_evidence="Initial seed",
        max_iterations=3
    )

    # Verify return types and values
    assert isinstance(best_action, str)
    assert isinstance(queries, list)
    assert len(queries) == 3
    
    # Verify the chosen best action is one of our candidate intents
    assert best_action in ["Analyze competitors", "Review patents", "Evaluate pricing"]
    assert queries == ["competitor pricing table", "competitor tier structure", "competitor discounts"]

    # Verify router was called correctly
    node_names_called = [call["node_name"] for call in mock_router.calls]
    assert "generate_candidate_intents" in node_names_called
    assert "simulate_outcome_state" in node_names_called
    assert "evaluate_state_quality" in node_names_called
    assert "generate_queries_for_intent" in node_names_called


@pytest.mark.asyncio
async def test_pcts_engine_repair_plan_success():
    """Test that repair_plan correctly generates alternative queries from the router response."""
    mock_router = MockLLMRouter()
    engine = PCTSEngine(router=mock_router)

    queries = await engine.repair_plan(
        failed_intent="Scrape competitor pricing",
        error_msg="Connection timed out",
        sub_question="Pricing models"
    )

    assert queries == ["repair query 1", "repair query 2", "repair query 3"]


@pytest.mark.asyncio
async def test_pcts_engine_repair_plan_failure_fallback():
    """Test that repair_plan falls back to the original sub-question on router failure."""
    mock_router = MockLLMRouter(fail_on=["repair_plan"])
    engine = PCTSEngine(router=mock_router)

    queries = await engine.repair_plan(
        failed_intent="Scrape competitor pricing",
        error_msg="Fatal error",
        sub_question="Pricing models"
    )

    assert queries == ["Pricing models"]


# ==========================================
# 3. Fallbacks and Edge Cases
# ==========================================

@pytest.mark.asyncio
async def test_pcts_engine_empty_evidence_state_quality():
    """Test that _evaluate_state_quality returns 0.1 instantly when evidence is empty."""
    engine = PCTSEngine(router=MockLLMRouter())
    score = await engine._evaluate_state_quality(
        sub_question="pricing",
        action="search",
        evidence=""
    )
    assert score == 0.1


@pytest.mark.asyncio
async def test_pcts_engine_evaluate_state_quality_regex_and_exceptions():
    """Test state quality parsing: regex patterns, invalid floats, and router exceptions."""
    # Test valid regex matching inside string
    mock_router = MockLLMRouter()
    engine = PCTSEngine(router=mock_router)
    
    # Override router response content dynamically for testing regex extraction
    async def custom_ainvoke(*args, **kwargs):
        return AIMessage(content="Score achieved is: 0.92, which is great.")
    
    mock_router.ainvoke = custom_ainvoke
    score = await engine._evaluate_state_quality("q", "a", "evidence")
    assert score == 0.92

    # Test non-numeric fallback
    async def custom_ainvoke_text(*args, **kwargs):
        return AIMessage(content="Excellent result without number")
    
    mock_router.ainvoke = custom_ainvoke_text
    score = await engine._evaluate_state_quality("q", "a", "evidence")
    assert score == 0.5

    # Test router exception fallback
    mock_router_fail = MockLLMRouter(fail_on=["evaluate_state_quality"])
    engine_fail = PCTSEngine(router=mock_router_fail)
    score_fail = await engine_fail._evaluate_state_quality("q", "a", "evidence")
    assert score_fail == 0.5


@pytest.mark.asyncio
async def test_pcts_engine_generate_candidate_intents_failure_fallback():
    """Test candidate intents fallback when the LLM router fails."""
    mock_router = MockLLMRouter(fail_on=["generate_candidate_intents"])
    engine = PCTSEngine(router=mock_router)

    intents = await engine._generate_candidate_intents(
        sub_question="How does X work?",
        topic="X analysis",
        current_evidence=""
    )
    assert intents == ["Investigate details of: How does X work?"]


@pytest.mark.asyncio
async def test_pcts_engine_simulate_outcome_state_failure_fallback():
    """Test outcome simulation fallback when the LLM router fails."""
    mock_router = MockLLMRouter(fail_on=["simulate_outcome_state"])
    engine = PCTSEngine(router=mock_router)

    outcome = await engine._simulate_outcome_state(
        sub_question="How does X work?",
        action="Ask X",
        evidence="previous facts"
    )
    assert outcome == {"evidence": "previous facts", "is_terminal": False}


@pytest.mark.asyncio
async def test_pcts_engine_generate_queries_for_intent_failure_fallback():
    """Test query variant generation fallback when the LLM router fails."""
    mock_router = MockLLMRouter(fail_on=["generate_queries_for_intent"])
    engine = PCTSEngine(router=mock_router)

    queries = await engine._generate_queries_for_intent(
        sub_question="How does X work?",
        intent="Scan documents"
    )
    assert queries == ["Scan documents", "How does X work?"]


@pytest.mark.asyncio
async def test_pcts_engine_search_no_child_plans_fallback():
    """Test search loop fallback behavior when zero iterations are performed and children list is empty."""
    mock_router = MockLLMRouter()
    engine = PCTSEngine(router=mock_router)

    best_action, queries = await engine.search(
        sub_question="Do we have any info?",
        topic="Empty search",
        initial_evidence="",
        max_iterations=0  # Prevents any iterations or child expansion
    )

    assert best_action == "Search for answers to: Do we have any info?"
    assert queries == ["Do we have any info?"]
