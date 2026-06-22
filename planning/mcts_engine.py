import asyncio
import math
import structlog
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from core.llm_router import LLMRouter

logger = structlog.get_logger("deep-research")

class MCTSNode:
    """A node in the Plan-Space Monte Carlo Tree Search."""
    
    def __init__(
        self,
        state: Dict[str, Any],
        action: Optional[str] = None,
        parent: Optional["MCTSNode"] = None,
        depth: int = 0
    ):
        self.state = state  # Tracks collected_evidence, remaining_gaps, etc.
        self.action = action  # Natural language intent/subplan used to get here
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_reward = 0.0
        self.depth = depth
        self.is_terminal = state.get("is_terminal", False) or depth >= 4
        self.unexpanded_actions: Optional[List[str]] = None

    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """Calculates Upper Confidence Bound Applied to Trees (UCT) score."""
        if self.visits == 0:
            return float("inf")
            
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_weight: float = 1.414) -> "MCTSNode":
        """Returns the child with the highest UCT score."""
        if not self.children:
            return self
        return max(self.children, key=lambda child: child.uct_score(exploration_weight))

    def update(self, reward: float):
        """Backpropagates visits and rewards upward."""
        self.visits += 1
        self.total_reward += reward


class PCTSEngine:
    """Plan-Space Monte Carlo Tree Search (MCTS) Engine for Advanced Research Planning."""
    
    def __init__(self, router: Optional[LLMRouter] = None):
        self.router = router or LLMRouter()

    async def search(
        self,
        sub_question: str,
        topic: str,
        initial_evidence: str = "",
        max_iterations: int = 3,
        exploration_weight: float = 1.414
    ) -> Tuple[str, List[str]]:
        """Executes Plan-Space MCTS to find the absolute best next research steps.
        
        Returns:
            Tuple[str, List[str]]: (Best next natural language intent, list of proposed queries/actions)
        """
        logger.info("Initializing Plan-MCTS Search", sub_question=sub_question, max_iterations=max_iterations)
        
        root_state = {
            "evidence": initial_evidence,
            "gaps": [sub_question],
            "is_terminal": False
        }
        root = MCTSNode(state=root_state, depth=0)
        
        for i in range(max_iterations):
            logger.info("MCTS Iteration", iteration=i+1, root_visits=root.visits)
            
            # 1. SELECTION
            node = root
            while node.children and not node.is_terminal:
                # If there are unexpanded actions, stop selecting and expand
                if node.unexpanded_actions:
                    break
                node = node.best_child(exploration_weight)
                
            # 2. EXPANSION
            if not node.is_terminal:
                if node.unexpanded_actions is None:
                    # Retrieve candidate next intents using the LLM
                    node.unexpanded_actions = await self._generate_candidate_intents(
                        sub_question=sub_question,
                        topic=topic,
                        current_evidence=node.state["evidence"]
                    )
                
                if node.unexpanded_actions:
                    chosen_action = node.unexpanded_actions.pop(0)
                    # Simulate outcome to get next state
                    next_state = await self._simulate_outcome_state(
                        sub_question=sub_question,
                        action=chosen_action,
                        evidence=node.state["evidence"]
                    )
                    child = MCTSNode(state=next_state, action=chosen_action, parent=node, depth=node.depth + 1)
                    node.children.append(child)
                    node = child  # Move to the expanded child for simulation
                    
            # 3. SIMULATION / EVALUATION
            reward = await self._evaluate_state_quality(
                sub_question=sub_question,
                action=node.action or "Initial Scoping",
                evidence=node.state["evidence"]
            )
            
            # 4. BACKPROPAGATION
            curr = node
            while curr is not None:
                curr.update(reward)
                curr = curr.parent

        # Choose the best action based on highest exploitation score
        if not root.children:
            logger.warning("MCTS generated no child plans, falling back to direct search")
            return f"Search for answers to: {sub_question}", [sub_question]
            
        best_node = max(root.children, key=lambda c: c.visits)
        logger.info("MCTS optimal path selected", best_action=best_node.action, reward_avg=best_node.total_reward / max(1, best_node.visits))
        
        # Generate refined search queries for the selected best action
        queries = await self._generate_queries_for_intent(sub_question, best_node.action)
        return best_node.action, queries

    async def repair_plan(
        self,
        failed_intent: str,
        error_msg: str,
        sub_question: str
    ) -> List[str]:
        """Dynamic Plan Repair: Recovers from scraping or search dead-ends by generating alternative strategies."""
        logger.warn("Dynamic Plan Repair triggered", failed_intent=failed_intent, error=error_msg)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert search planner. A research search intent has failed due to connection issues or empty results.\n"
                "Your job is to dynamically REPAIR the plan by generating 3 alternative, highly robust search queries that bypass the failure.\n"
                "For example, use broader terms, synonymous concepts, alternative public sources, or focus on adjacent details.\n"
                "CRITICAL: Do NOT start any search query with conversational or instructional verbs (e.g., 'explain', 'describe', 'define', 'discuss', 'analyze', 'summarize', 'find', 'search', 'get', 'what is', 'how to'). Standard search engines interpret these as instructions to find dictionary/glossary definitions, leading to low-quality, generic results. Instead, output direct nouns, technical terms, specifications, and factual keyword phrases."
            )),
            ("user", (
                "Target Research Question: {sub_question}\n"
                "Failed Search Intent: {failed_intent}\n"
                "Error encountered: {error_msg}\n\n"
                "Provide exactly 3 alternative search queries, one per line. No markdown tags, no numbering, just the queries."
            ))
        ])
        
        try:
            response = await self.router.ainvoke(
                messages=prompt.format_messages(
                    sub_question=sub_question,
                    failed_intent=failed_intent,
                    error_msg=error_msg
                ),
                tier="STANDARD",
                agent_name="MCTSPlanner",
                node_name="repair_plan"
            )
            queries = [q.strip("- ").strip() for q in response.content.strip().split("\n") if q.strip()]
            return queries[:3]
        except Exception as e:
            logger.error("Failed to generate plan repair queries, falling back to direct query", error=str(e))
            return [sub_question]

    # LLM-assisted Simulation, Expansion, and Evaluation APIs

    async def _generate_candidate_intents(
        self,
        sub_question: str,
        topic: str,
        current_evidence: str
    ) -> List[str]:
        """Generates high-level natural language search intents / subplans."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an analyst-grade planning engine. Based on the target research sub-question, topic, and current evidence gathered, "
                "identify 3 distinct, high-level next natural-language research intents (subplans) to explore. "
                "Each intent should be a specific direction of inquiry (e.g., 'Analyze competitor pricing tables' or 'Retrieve clinical trial efficacy rates').\n"
                "Provide exactly 3 intents, one per line, no numbering, no markdown formatting."
            )),
            ("user", (
                "Research Topic: {topic}\n"
                "Research Question: {sub_question}\n\n"
                "Current Evidence Gained:\n{evidence}\n\n"
                "Suggest 3 high-level next steps:"
            ))
        ])
        
        try:
            response = await self.router.ainvoke(
                messages=prompt.format_messages(
                    topic=topic,
                    sub_question=sub_question,
                    evidence=current_evidence or "No evidence collected yet."
                ),
                tier="STANDARD",
                agent_name="MCTSPlanner",
                node_name="generate_candidate_intents"
            )
            intents = [line.strip().strip("- ") for line in response.content.strip().split("\n") if line.strip()]
            return intents[:3]
        except Exception as e:
            logger.error("Failed to generate candidate intents", error=str(e))
            return [f"Investigate details of: {sub_question}"]

    async def _simulate_outcome_state(
        self,
        sub_question: str,
        action: str,
        evidence: str
    ) -> Dict[str, Any]:
        """Simulates the state transition (mental simulation of evidence expansion)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an evaluator simulating research progress. Predict how the available research state "
                "will change if we execute the proposed research action. Highlight what new facts are likely to be uncovered "
                "relative to the target question.\n"
                "Summarize the simulated updated evidence in a short, concise paragraph."
            )),
            ("user", (
                "Target Question: {sub_question}\n"
                "Current Evidence: {evidence}\n"
                "Proposed Research Step: {action}\n\n"
                "Simulated Evidence Gained:"
            ))
        ])
        
        try:
            response = await self.router.ainvoke(
                messages=prompt.format_messages(
                    sub_question=sub_question,
                    evidence=evidence or "None",
                    action=action
                ),
                tier="STANDARD",
                agent_name="MCTSPlanner",
                node_name="simulate_outcome_state"
            )
            new_evidence = f"{evidence}\nSimulated Info for [{action}]: {response.content.strip()}"
            is_terminal = "satisfied" in response.content.lower() or "fully answered" in response.content.lower()
            return {
                "evidence": new_evidence,
                "is_terminal": is_terminal
            }
        except Exception as e:
            logger.error("Failed to simulate outcome state", error=str(e))
            return {"evidence": evidence, "is_terminal": False}

    async def _evaluate_state_quality(
        self,
        sub_question: str,
        action: str,
        evidence: str
    ) -> float:
        """Heuristic/LLM reward evaluator. Evaluates the quality/density of evidence gathered (Returns R in [0, 1])."""
        if not evidence:
            return 0.1
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an objective auditor scoring research findings. Rate how effectively the available evidence "
                "uncovers facts and answers the target sub-question. Give a single numerical score between 0.0 (completely useless/unrelated) "
                "and 1.0 (perfect, comprehensive answer).\n"
                "Respond with ONLY the numerical score. Do not explain your score."
            )),
            ("user", (
                "Target Question: {sub_question}\n"
                "Evidence Collected:\n{evidence}\n\n"
                "Score (0.0 to 1.0):"
            ))
        ])
        
        try:
            response = await self.router.ainvoke(
                messages=prompt.format_messages(
                    sub_question=sub_question,
                    evidence=evidence
                ),
                tier="STANDARD",
                agent_name="MCTSPlanner",
                node_name="evaluate_state_quality"
            )
            score_str = response.content.strip()
            # Attempt to parse float
            import re
            match = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", score_str)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            logger.error("Failed to evaluate state quality, falling back to default score", error=str(e))
            return 0.5

    async def _generate_queries_for_intent(self, sub_question: str, intent: str) -> List[str]:
        """Generates concrete web search queries to execute a specific high-level intent."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a search query optimizer. Given a target research sub-question and a chosen plan intent, "
                "generate exactly 3 highly specific, diverse web search queries designed to gather the precise evidence needed.\n"
                "CRITICAL: Do NOT start any search query with conversational or instructional verbs (e.g., 'explain', 'describe', 'define', 'discuss', 'analyze', 'summarize', 'find', 'search', 'get', 'what is', 'how to'). Standard search engines interpret these as instructions to find dictionary/glossary definitions, leading to low-quality, generic results. Instead, output direct nouns, technical terms, specifications, and factual keyword phrases.\n"
                "Provide exactly 3 queries, one per line. No numbering, no markdown formatting."
            )),
            ("user", (
                "Target Question: {sub_question}\n"
                "Plan Intent: {intent}\n\n"
                "Search Queries:"
            ))
        ])
        
        try:
            response = await self.router.ainvoke(
                messages=prompt.format_messages(
                    sub_question=sub_question,
                    intent=intent
                ),
                tier="STANDARD",
                agent_name="MCTSPlanner",
                node_name="generate_queries_for_intent"
            )
            queries = [q.strip("- ").strip() for q in response.content.strip().split("\n") if q.strip()]
            return queries[:3]
        except Exception as e:
            logger.error("Failed to generate queries for intent", error=str(e))
            return [intent, sub_question]
