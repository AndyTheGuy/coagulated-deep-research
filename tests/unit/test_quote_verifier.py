import pytest
from core.models import Claim, VerifiedSource, QuoteVerification
from verification.quote_verifier import QuoteVerifier

def test_quote_verifier_exact_match():
    verifier = QuoteVerifier()
    doc_content = "This is a comprehensive study showing that Company X grew revenue by 15 percent in Q3 2026."
    
    # Exact match with capitalization differences and whitespace normalization
    quote = "  company X grew revenue by 15 percent   "
    
    result = verifier.verify_quote(quote, doc_content)
    
    assert isinstance(result, QuoteVerification)
    assert result.quote == quote
    assert result.is_verified is True
    assert result.score == 1.0
    assert result.matched_text == "Company X grew revenue by 15 percent"

def test_quote_verifier_fuzzy_match_pass():
    verifier = QuoteVerifier(threshold=0.85)
    doc_content = "The research team found that NIST expects full deployment of Post-Quantum Cryptography algorithms by the year 2028."
    
    # Fuzzy match with slight variations (e.g., "PQC" vs "Post-Quantum Cryptography" is too different, but small wording edits pass)
    quote = "NIST expects full deployment of Post Quantum Cryptography by 2028"
    
    result = verifier.verify_quote(quote, doc_content)
    
    assert isinstance(result, QuoteVerification)
    assert result.is_verified is True
    assert result.score >= 0.85
    assert result.matched_text is not None
    assert "NIST expects full deployment" in result.matched_text

def test_quote_verifier_fuzzy_match_fail():
    verifier = QuoteVerifier(threshold=0.85)
    doc_content = "In 2026, the global economy recovered slowly."
    quote = "In 2028, the Mars colony was successfully established."
    
    result = verifier.verify_quote(quote, doc_content)
    
    assert isinstance(result, QuoteVerification)
    assert result.is_verified is False
    assert result.score < 0.85
    assert result.matched_text is None

def test_quote_verifier_empty_inputs():
    verifier = QuoteVerifier()
    
    res1 = verifier.verify_quote("", "some document")
    assert res1.is_verified is False
    assert res1.score == 0.0
    
    res2 = verifier.verify_quote("some quote", "")
    assert res2.is_verified is False
    assert res2.score == 0.0

def test_verify_claim_quotes():
    verifier = QuoteVerifier()
    
    claim = Claim(
        claim_id="c1",
        claim_text="Revenue grew by 15%",
        section="Finance",
        supporting_quotes=["revenue by 15 percent", "failed quote about Mars"]
    )
    
    sources = [
        VerifiedSource(
            url="https://finance.com",
            title="Finance Report",
            content="This shows that Company X increased revenue by 15 percent in Q3.",
            accessible=True,
            status_code=200
        ),
        VerifiedSource(
            url="https://bad.com",
            title="Inaccessible",
            content="",
            accessible=False,
            status_code=404
        )
    ]
    
    verifications = verifier.verify_claim_quotes(claim, sources)
    
    assert len(verifications) == 2
    assert verifications[0].is_verified is True
    assert verifications[0].score == 1.0
    assert verifications[1].is_verified is False
    assert verifications[1].score < 0.5
