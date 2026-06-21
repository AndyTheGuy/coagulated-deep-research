import pytest
from langgraph.types import Send
from core.models import GraphState, ResearchBrief, SubQuestion, ResearcherInput, reduce_sub_questions
from core.router import supervisor_node, route_research

def test_reduce_sub_questions():
    left = [
        SubQuestion(id="q1", question="What is A?", status="pending"),
        SubQuestion(id="q2", question="What is B?", status="pending"),
    ]
    right = [
        SubQuestion(id="q1", question="What is A?", status="completed", assigned_researcher="agent-1", results_summary="A is apple"),
    ]
    
    merged = reduce_sub_questions(left, right)
    
    assert len(merged) == 2
    q1 = next(q for q in merged if q.id == "q1")
    q2 = next(q for q in merged if q.id == "q2")
    
    assert q1.status == "completed"
    assert q1.assigned_researcher == "agent-1"
    assert q1.results_summary == "A is apple"
    
    assert q2.status == "pending"
    assert q2.assigned_researcher is None

@pytest.mark.asyncio
async def test_supervisor_node_initialization():
    brief = ResearchBrief(
        topic="AI Benchmarks",
        scope="Benchmarks comparison",
        sub_questions=[
            SubQuestion(id="q1", question="What is A?"),
            SubQuestion(id="q2", question="What is B?"),
        ]
    )
    
    state = GraphState(
        topic="AI Benchmarks",
        research_brief=brief,
        sub_questions_state=[] # Empty sub_questions_state
    )
    
    updates = await supervisor_node(state)
    
    # Should update/initialize sub_questions_state from research brief
    assert "sub_questions_state" in updates
    assert len(updates["sub_questions_state"]) == 2
    assert updates["sub_questions_state"][0].id == "q1"

@pytest.mark.asyncio
async def test_supervisor_node_no_op_when_populated():
    brief = ResearchBrief(
        topic="AI Benchmarks",
        scope="Benchmarks comparison",
        sub_questions=[SubQuestion(id="q1", question="What is A?")]
    )
    
    state = GraphState(
        topic="AI Benchmarks",
        research_brief=brief,
        sub_questions_state=[SubQuestion(id="q1", question="What is A?", status="in_progress")]
    )
    
    updates = await supervisor_node(state)
    
    # Should be no-op if sub_questions_state is already populated
    assert updates == {}

def test_route_research_pending():
    brief = ResearchBrief(
        topic="Quantum Computing",
        scope="Quantum key distribution details",
        constraints=["use arXg"],
        sub_questions=[
            SubQuestion(id="q1", question="What is QKD?", status="pending"),
            SubQuestion(id="q2", question="Who invented QKD?", status="completed", results_summary="Charles Bennett"),
        ]
    )
    
    state = GraphState(
        topic="Quantum Computing",
        research_brief=brief,
        sub_questions_state=[
            SubQuestion(id="q1", question="What is QKD?", status="pending"),
            SubQuestion(id="q2", question="Who invented QKD?", status="completed", results_summary="Charles Bennett"),
        ]
    )
    
    sends = route_research(state)
    
    # Should spawn researcher_node strictly for pending questions (q1)
    assert isinstance(sends, list)
    assert len(sends) == 1
    assert isinstance(sends[0], Send)
    assert sends[0].node == "researcher_node"
    
    payload = sends[0].arg
    assert isinstance(payload, ResearcherInput)
    assert payload.sub_question.id == "q1"
    assert payload.topic == "Quantum Computing"
    assert payload.constraints == ["use arXg"]

def test_route_research_all_completed():
    brief = ResearchBrief(
        topic="Quantum Computing",
        scope="Quantum key distribution details",
        sub_questions=[SubQuestion(id="q1", question="What is QKD?")]
    )
    
    state = GraphState(
        topic="Quantum Computing",
        research_brief=brief,
        sub_questions_state=[SubQuestion(id="q1", question="What is QKD?", status="completed", results_summary="Detailed explanation")]
    )
    
    route = route_research(state)
    
    # Should route to context_aggregator if none are pending
    assert route == "context_aggregator"

def test_route_research_no_questions():
    state = GraphState(topic="Empty Research", sub_questions_state=[])
    route = route_research(state)
    assert route == "context_aggregator"
