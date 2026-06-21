import operator
from core.models import (
    SubQuestion,
    ResearchBrief,
    SearchResult,
    VerifiedSource,
    Claim,
    Report,
    GraphState,
    QuoteVerification,
    VerificationResult,
    ReportConfidenceScore,
    merge_dict_reducer
)

def test_sub_question_defaults():
    """Verify default values and custom fields for SubQuestion."""
    sq = SubQuestion(id="q1", question="What is quantum key distribution?")
    assert sq.id == "q1"
    assert sq.question == "What is quantum key distribution?"
    assert sq.status == "pending"
    assert sq.assigned_researcher is None
    assert sq.results_summary is None

def test_research_brief_validation():
    """Verify ResearchBrief handles lists and sub-objects correctly."""
    sq = SubQuestion(id="q1", question="What is QKD?")
    brief = ResearchBrief(
        topic="Quantum Cryptography",
        scope="Study QKD and post-quantum algorithms",
        sub_questions=[sq],
        constraints=["use verified sources only"]
    )
    assert brief.topic == "Quantum Cryptography"
    assert len(brief.sub_questions) == 1
    assert brief.sub_questions[0].id == "q1"
    assert brief.target_source_count == 20

def test_search_result_fields():
    """Verify SearchResult parses correctly."""
    sr = SearchResult(
        title="Speedtest",
        url="https://speedtest.net",
        content="Ookla Speedtest",
        score=0.95,
        engine="google"
    )
    assert sr.title == "Speedtest"
    assert sr.url == "https://speedtest.net"
    assert sr.score == 0.95
    assert sr.engine == "google"

def test_claim_defaults():
    """Verify Claim default initialization."""
    claim = Claim(claim_id="c1", claim_text="QKD is secure", section="Security")
    assert claim.claim_id == "c1"
    assert claim.verification_status == "unverified"
    assert claim.confidence_score == 0.0
    assert len(claim.supporting_quotes) == 0

def test_report_compilation():
    """Verify Report model parses claims and citation structures."""
    claim = Claim(claim_id="c1", claim_text="QKD is secure", section="Security")
    report = Report(
        title="Quantum Report",
        content="# Quantum Report\nSecure transmission...",
        claims=[claim],
        citations=["https://speedtest.net"],
        confidence_score=0.85
    )
    assert report.title == "Quantum Report"
    assert len(report.claims) == 1
    assert report.claims[0].claim_id == "c1"
    assert report.citations == ["https://speedtest.net"]
    assert report.confidence_score == 0.85

def test_graph_state_defaults_and_reducers():
    """Verify GraphState fields and custom dict/list aggregation."""
    # Instantiation
    state = GraphState(
        topic="Quantum Key Distribution",
        user_query="How does quantum key distribution work?"
    )
    assert state.topic == "Quantum Key Distribution"
    assert state.clarification_needed is False
    assert len(state.search_results) == 0
    assert len(state.token_usage) == 0

    # Test merge_dict_reducer
    d1 = {
        "vertex_ai": {"input_tokens": 100, "output_tokens": 50, "calls": 2},
        "failovers": 1
    }
    d2 = {
        "vertex_ai": {"input_tokens": 50, "output_tokens": 25, "calls": 1},
        "freellmapi": {"input_tokens": 200, "output_tokens": 100, "calls": 4},
        "failovers": 2
    }
    merged = merge_dict_reducer(d1, d2)
    assert merged == {
        "vertex_ai": {"input_tokens": 150, "output_tokens": 75, "calls": 3},
        "freellmapi": {"input_tokens": 200, "output_tokens": 100, "calls": 4},
        "failovers": 3
    }

    # Test operator.add on list fields in GraphState context
    l1 = [SearchResult(title="t1", url="u1", content="c1")]
    l2 = [SearchResult(title="t2", url="u2", content="c2")]
    combined = operator.add(l1, l2)
    assert len(combined) == 2
    assert combined[0].title == "t1"
    assert combined[1].title == "t2"

def test_pydantic_validation_constraints():
    """Verify that Pydantic models enforce enums and range constraints."""
    import pytest
    from pydantic import ValidationError
    
    # Invalid SubQuestion status
    with pytest.raises(ValidationError):
        SubQuestion(id="q1", question="test", status="invalid_status")
        
    # Invalid Claim verification_status
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", claim_text="test", section="test", verification_status="invalid_status")
        
    # Invalid Claim confidence_score (too low)
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", claim_text="test", section="test", confidence_score=-0.1)
        
    # Invalid Claim confidence_score (too high)
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", claim_text="test", section="test", confidence_score=1.1)

def test_new_models_defaults_and_validation():
    """Verify initialization, default values, and structure of the new Phase 3 models."""
    import pytest
    from pydantic import ValidationError

    # 1. QuoteVerification
    qv = QuoteVerification(quote="He told them so", is_verified=True, score=0.92, matched_text="He told them so.")
    assert qv.quote == "He told them so"
    assert qv.is_verified is True
    assert qv.score == 0.92
    assert qv.matched_text == "He told them so."

    # 2. VerificationResult
    vr = VerificationResult(
        claim_id="c123",
        claim_text="The speed of light is constant",
        status="verified",
        confidence_score=0.95,
        quotes_verification=[qv],
        source_status={"https://wikipedia.org": True}
    )
    assert vr.claim_id == "c123"
    assert vr.status == "verified"
    assert vr.confidence_score == 0.95
    assert len(vr.quotes_verification) == 1
    assert vr.source_status["https://wikipedia.org"] is True

    # 3. ReportConfidenceScore
    rcs = ReportConfidenceScore(
        overall_score=0.88,
        verified_claims_count=8,
        total_claims_count=10,
        unverified_claims_count=1,
        failed_claims_count=1,
        gaps_count=0
    )
    assert rcs.overall_score == 0.88
    assert rcs.verified_claims_count == 8
    assert rcs.total_claims_count == 10

    # Test Validation constraints
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="c123",
            claim_text="test",
            status="invalid_status",  # Must be verified, unverified, failed, gap
            confidence_score=0.5
        )
