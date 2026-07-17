
import os
import re
from typing import TypedDict, List, Optional, Literal
from datetime import datetime, timezone
AUDIT_LOG_PATH = "audit_log.jsonl"
HUMAN_REVIEW_QUEUE_PATH = "human_review_queue.jsonl"

from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END



# ---------- Model configuration ----------
GRADER_MODEL = "llama-3.1-8b-instant"  #"openai/gpt-oss-120b"
DECISION_MODEL ="llama-3.3-70b-versatile"  #"openai/gpt-oss-120b"

TOKEN_LIMITS = {
    "build_query": 100,
    "grade_relevance": 250,
    "rewrite_query": 100,
    "decision": 600,
    "grounding_check": 350,
}

PROMPT_VERSION = "claims-agent-prompts-v2.0"
MAX_RETRIEVAL_RETRIES_CAP = 2
MAX_DECISION_RETRIES = 1


# ---------- Sanitization ----------
def sanitize_text(text: str) -> str:
    """
    Remove all non-ASCII characters from the text.
    Ensures no special characters (emojis, smart quotes, etc.) cause encoding errors or tool choice crashes.
    """
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


# ---------- Schemas ----------
class ClaimInput(BaseModel):
    """Structured, validated representation of an incoming claim request."""
    claim_query: str = Field(..., min_length=10, max_length=2000,
                             description="Free-text description of what happened and what coverage is being asked about")
    policy_type: Literal["auto"] = Field(
        default="auto", description="Works best for Auto Insurance")
    policy_number: Optional[str] = Field(default=None, max_length=50)
    claimant_name: Optional[str] = Field(default=None, max_length=200)
    date_of_loss: Optional[str] = Field(default=None, description="Date of loss, if known (any readable format)")
    claimed_amount: Optional[float] = Field(default=None, ge=0, description="Dollar amount being claimed, if known")

    @field_validator("claim_query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim_query cannot be blank")
        return sanitize_text(v.strip())


class RelevanceGrade(BaseModel):
    relevant: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )
    official_source: bool
    reasoning: str


class Decision(BaseModel):
    verdict: Literal["approve", "deny", "escalate"]

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the final decision"
    )

    reasoning: str = Field(
        ...,
        min_length=10,
        description="Clear explanation based only on retrieved evidence"
    )

    cited_clause_ids: List[str] = Field(
        default_factory=list,
        description="Clause IDs supporting the decision"
    )

    missing_information: List[str] = Field(
        default_factory=list,
        description="Missing claim details or evidence that prevent a confident decision"
    )

    recommended_action: str = Field(
        ...,
        description="Next action such as approve payment, deny claim, or send for human review"
    )

class HallucinationGrade(BaseModel):
    grounded: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    citation_ids_valid: bool
    verdict_supported: bool

    unsupported_claims: List[str] = Field(
        default_factory=list
    )

    ignored_conflicts: List[str] = Field(
        default_factory=list
    )

    reasoning: str


class AuditRecord(BaseModel):
    claim_id: str
    claim_query: str
    verdict: Literal["approve", "deny", "escalate"]
    reasoning: str
    relevance_confidence: Optional[float]
    hallucination_confidence: Optional[float]
    cited_clause_ids: List[str]
    retrieved_clause_ids: List[str]
    search_queries_tried: List[str]
    retry_count: int
    web_sources_consulted: List[dict] = []
    requires_human_review: bool
    escalation_priority: Optional[str] = None
    model_name_grader: str
    model_name_decision: str
    prompt_version: str
    timestamp: str


class ClaimState(TypedDict):
    claim_id: str
    raw_claim_input: dict
    claim_input: Optional[dict]
    validation_error: Optional[str]
    claim_query: str
    search_query: str
    query_history: List[str]
    vectorstore: object
    retrieved_docs: List[Document]
    relevance_grade: Optional[RelevanceGrade]
    retry_count: int
    max_retries: int
    relevance_threshold: float
    decision: Optional[Decision]
    hallucination_grade: Optional[HallucinationGrade]
    decision_retry_count: int
    hallucination_threshold: float
    audit_record: Optional[dict]
    enable_web_fallback: bool
    tavily_client: object
    web_search_results: List[dict]
    verified_regulatory_results: List[dict]


# ---------- System prompts ----------
BUILD_QUERY_SYSTEM_PROMPT = """You are preparing the first search query against an insurance policy database for a newly submitted claim.

Policy type (if known): {policy_type}
Claim scenario: {claim_query}

Write ONE concise search query using terms likely to appear in policy documents
(coverage, exclusion, section names) rather than conversational language.
Return ONLY the query text, nothing else."""

RELEVANCE_GRADER_SYSTEM_PROMPT = """
You are evaluating whether the retrieved insurance policy material is
sufficiently relevant to help make a claims decision.

Claim:
{claim_query}

Retrieved policy material:
{retrieved_context}

The retrieved document may be:
- a declarations page,
- full policy wording,
- an endorsement,
- a coverage schedule,
- or a policy summary.

Evaluate separately:

1. Does the material address the cause of loss?
2. Does it identify an applicable coverage, deductible, limit, or endorsement?
3. Does it contain a relevant exclusion or limitation?
4. Does it contain an exception, endorsement, or definition that may alter coverage?
5. Does it contain conditions, duties, or evidence requirements?
6. Does it materially help support approve, deny, or escalate?

DOCUMENT RELEVANCE RULES:

- A declarations page is relevant when it lists a coverage, deductible,
  limit, insured vehicle, policy period, or endorsement connected to the claim.

- Do not mark a declarations page irrelevant merely because it does not
  contain full contractual wording such as "we will pay."

- For a straightforward collision, comprehensive, glass, rental,
  medical-payments, uninsured-motorist, or OEM-parts claim, a matching listed
  coverage counts as meaningful retrieval evidence.

- Full exclusion language is still required before the material can support
  a denial.

- If the material identifies applicable coverage but lacks the exclusions
  or conditions needed for a final decision, mark it relevant and explain
  that the evidence may still be incomplete.

- A document is not sufficiently relevant merely because it mentions the same
  general topic.

Set relevant=true when the retrieved material materially helps determine
coverage, exclusion, limitation, endorsement, condition, or escalation.

Use only the supplied text.
"""

REWRITE_SYSTEM_PROMPT = """You are refining a search query against an insurance policy database.
The previous search attempt did not retrieve clauses that clearly resolve the claim scenario.

Original claim scenario: {claim_query}
Previous search query: {search_query}
Why the previous retrieval was weak: {grade_reasoning}

Write ONE improved search query, phrased using terms likely to appear in policy documents.
Return ONLY the query text, nothing else."""

DECISION_SYSTEM_PROMPT = """
You are a conservative insurance claims adjudication assistant.

Your task is to evaluate the submitted claim using ONLY the policy clauses
and verified regulatory material supplied in the context.

Do not use general insurance knowledge, assumptions, industry customs,
or information that does not appear in the supplied evidence.

CLAIM:
{claim_query}

RETRIEVED POLICY CLAUSES:
{retrieved_context}

VERIFIED REGULATORY CONTEXT:
{regulatory_context}

Follow this analysis order exactly:

1. POLICY APPLICABILITY
    Determine whether the supplied clauses establish that the policy applies
    to the claimant, insured property, incident date, driver, location,
    and vehicle use.

2. COVERAGE GRANT
    Identify whether a specific clause affirmatively provides coverage
    for the cause of loss and type of damage described in the claim.

3. EXCLUSIONS AND LIMITATIONS
    Identify any clause that explicitly excludes, limits, suspends,
    or removes that coverage.

4. ENDORSEMENTS AND EXCEPTIONS
    Determine whether any endorsement, exception, or special provision
    restores or changes coverage.

5. CONDITIONS AND DUTIES
    Check whether the policy requires reporting, documentation,
    cooperation, police reports, inspections, estimates, or other evidence.

6. FACT SUFFICIENCY
    Determine whether the facts required to apply the relevant clauses
    are clearly present in the claim.

DOCUMENT INTERPRETATION RULES:

- The retrieved document may be a declarations page, full policy wording,
  endorsement, or a combination of these.

- If a declarations page clearly lists an active coverage, deductible,
  limit, or endorsement, treat that as evidence that the coverage or
  benefit is included in the policy.

- For a straightforward claim that directly matches a listed coverage,
  APPROVE may be selected when no retrieved exclusion, limitation,
  conflict, or missing critical fact defeats coverage.

- Do not require a declarations page to contain full contract phrases
  such as "we will pay" when the coverage is clearly listed.

- Do not use a declarations page to invent exclusions that are not shown.

- If denial depends on an exclusion, condition, definition, or endorsement
  that was not retrieved, ESCALATE instead of assuming it applies.

Decision rules:

- APPROVE when:
  1. The declarations page or policy wording clearly shows the applicable
     coverage is included.
  2. The claim facts directly match that coverage.
  3. No retrieved exclusion, limitation, or conflicting clause defeats coverage.
  4. The essential claim facts are available.

- DENY only when:
  1. A retrieved clause explicitly excludes, limits, suspends, or removes coverage.
  2. The exclusion clearly applies to the stated facts.
  3. No retrieved endorsement or exception restores coverage.

- ESCALATE when:
  1. The applicable coverage cannot be identified.
  2. Denial would require an exclusion that was not retrieved.
  3. Important claim facts are missing or conflicting.
  4. Coverage depends on an unavailable endorsement, definition, or condition.
  5. The retrieved policy documents conflict.
  6. The policy and verified regulatory material appear to conflict.

Important restrictions:

- Silence is not an exclusion.
- Do not deny merely because affirmative coverage was not retrieved.
- Do not assume that an expired policy, excluded driver, rideshare use,
    late reporting, missing evidence, fraud, or intoxication results in denial
    unless a supplied clause explicitly states the relevant consequence.
- Do not invent deductibles, limits, dates, exclusions, definitions,
    endorsements, conditions, or legal requirements.
- Every material statement in the reasoning must be traceable to a retrieved clause.
- Cite only clause IDs that appear in the retrieved context.
- If a cited clause does not directly support the statement, escalate.
- When uncertain, escalate instead of guessing.

In the reasoning:
- State the applicable coverage clause.
- State any applicable exclusion or limitation.
- State whether an endorsement or exception applies.
- Identify missing facts or documents.
- Explain why the evidence supports the selected verdict.

Return one structured Decision object with:

- verdict:
  "approve", "deny", or "escalate"

- confidence:
  A number from 0.0 to 1.0 representing confidence in the decision.
  Use lower confidence when policy evidence or claim facts are incomplete.

- reasoning:
  A concise explanation grounded only in the supplied evidence.

- cited_clause_ids:
  Only clause IDs that appear in the retrieved policy context.
  Use an empty list when no clause can safely be cited.

- missing_information:
  List any facts, clauses, endorsements, or documents required for a safer
  decision. Use an empty list when nothing important is missing.

- recommended_action:
  State the appropriate next step, such as:
  "Approve the covered claim subject to applicable deductible",
  "Deny based on the cited exclusion",
  or
  "Send to a human adjuster for further review".
"""

HALLUCINATION_GRADER_SYSTEM_PROMPT = """
You are auditing an insurance claim decision.

Verify the decision using only the retrieved policy clauses and verified
regulatory context.

Check all of the following:

1. Every cited clause ID exists in the retrieved context.
2. Every factual policy statement is supported by a cited clause.
3. The verdict logically follows from those clauses.
4. An approval has an affirmative coverage grant.
5. A denial has an explicit exclusion, limitation, or applicable condition.
6. No conflicting clause, endorsement, or exception was ignored.
7. No missing fact was silently assumed.
8. No deductible, limit, date, definition, exclusion, or legal rule was invented.
9. Regulatory material was not used to rewrite the contract unless it clearly
    applies and comes from a verified authoritative source.

Set grounded=true only when both the reasoning and the verdict are fully supported.

GROUNDING RULES:

- Do not fail an APPROVE decision merely because the retrieved document is
  a declarations page rather than full policy wording.

- A listed coverage, deductible, limit, or endorsement on a declarations page
  may support a straightforward approval when the claim directly matches it.

- Fail the decision only when it invents or contradicts:
  coverage,
  exclusions,
  deductibles,
  limits,
  endorsements,
  dates,
  or legal requirements.

- An ESCALATE decision should be considered grounded when it correctly states
  that the available policy evidence is incomplete, ambiguous, or insufficient.

- Do not mark ordinary conservative reasoning as hallucination when it is clearly
  based on the retrieved document.
Otherwise identify:
- unsupported claims,
- invalid citations,
- ignored conflicts,
- missing facts,
- and the safest corrective action.
"""

AUTHORITATIVE_INSURANCE_DOMAINS = ["naic.org", "content.naic.org"]


# ---------- Ingestion ----------
def build_vectorstore_from_files(file_paths: List[str]) -> Chroma:
    raw_docs = []

    for path in file_paths:
        loader = (
            PyPDFLoader(path)
            if path.lower().endswith(".pdf")
            else TextLoader(path)
        )

        docs = loader.load()

        for doc in docs:
            doc.page_content = sanitize_text(doc.page_content)

        raw_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        source = os.path.basename(
            chunk.metadata.get("source", "unknown")
        )

        page = chunk.metadata.get("page", 0)

        chunk.metadata["clause_id"] = (
            f"{source}::page_{page + 1}::chunk_{i}"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="insurance_policy_clauses"
    )


def format_docs_for_grading(docs):
    return "\n\n".join(f"[{d.metadata['clause_id']}]\n{d.page_content}" for d in docs)

def format_claim(claim_input: dict) -> str:
    return (
        f"Claim description: {claim_input.get('claim_query', '')}\n"
        f"Policy type: {claim_input.get('policy_type', 'unknown')}\n"
        f"Policy number: {claim_input.get('policy_number', 'unknown')}\n"
        f"Claimant: {claim_input.get('claimant_name', 'unknown')}\n"
        f"Date of loss: {claim_input.get('date_of_loss', 'unknown')}\n"
        f"Claimed amount: {claim_input.get('claimed_amount', 'unknown')}"
    )


def format_web_results(results: List[dict]) -> str:
    if not results:
        return "No verified regulatory context."

    return "\n\n".join(
        f"Title: {result.get('title', '')}\n"
        f"URL: {result.get('url', '')}\n"
        f"Extract: {result.get('snippet', '')}"
        for result in results
    )

def _escalation_priority(claim_input: Optional[dict]) -> str:
    if claim_input and claim_input.get("claimed_amount") is not None and claim_input["claimed_amount"] >= 25000:
        return "high"
    return "normal"


# ---------- Graph builder ----------
def build_claims_graph(groq_api_key: str):
    build_query_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["build_query"], api_key=groq_api_key)
    grader_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["grade_relevance"], api_key=groq_api_key)
    rewrite_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["rewrite_query"], api_key=groq_api_key)
    grounding_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["grounding_check"], api_key=groq_api_key)
    strong_llm = ChatGroq(model=DECISION_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["decision"], api_key=groq_api_key)

    build_query_chain = ChatPromptTemplate.from_messages([("system", BUILD_QUERY_SYSTEM_PROMPT)]) | build_query_llm

    # Method function_calling explicitly enforces the model uses the tool
    relevance_grader_chain = (
        ChatPromptTemplate.from_messages([("system", RELEVANCE_GRADER_SYSTEM_PROMPT)])
        | grader_llm.with_structured_output(RelevanceGrade, method="function_calling")
    )
    rewrite_chain = ChatPromptTemplate.from_messages([("system", REWRITE_SYSTEM_PROMPT)]) | rewrite_llm
    decision_chain = (
        ChatPromptTemplate.from_messages([("system", DECISION_SYSTEM_PROMPT)])
        | strong_llm.with_structured_output(Decision, method="function_calling")
    )
    hallucination_chain = (
        ChatPromptTemplate.from_messages([
            ("system", HALLUCINATION_GRADER_SYSTEM_PROMPT),
            (
                "human",
                """
    Retrieved policy material:
    {retrieved_context}

    Decision verdict:
    {verdict}

    Decision reasoning:
    {decision_reasoning}

    Cited clause IDs:
    {cited_clause_ids}

    Evaluate whether this specific decision is grounded in the retrieved material.
    Return exactly one valid JSON object with these fields:
    grounded, confidence, citation_ids_valid, verdict_supported,
    unsupported_claims, ignored_conflicts, reasoning.

    Do not repeat any field.
    """
            )
        ])
        | grounding_llm.with_structured_output(
            HallucinationGrade,
            method="json_mode"
        )
    )

    # ----- Nodes -----
    def validate_claim_node(state):
        try:
            # Sanitize dictionary values
            clean_input = {k: sanitize_text(str(v)) if isinstance(v, str) else v for k, v in state["raw_claim_input"].items()}
            claim_input = ClaimInput(**clean_input)
            return {
                "claim_input": claim_input.model_dump(),
                "claim_query": claim_input.claim_query,
                "validation_error": None,
            }
        except ValidationError as e:
            return {"claim_input": None, "validation_error": str(e)}

    def invalid_input_node(state):
        reasoning = f"Claim input failed validation and cannot be processed automatically: {state['validation_error']}"
        return {
    "decision": Decision(verdict="escalate", confidence=1.0, reasoning=reasoning, cited_clause_ids=[], missing_information=["Valid structured claim input"], recommended_action="Send the claim to a human reviewer for input correction.")}

    def build_query_node(state):
        claim_input = state["claim_input"] or {}
        query = build_query_chain.invoke({
            "policy_type": claim_input.get("policy_type") or "unknown",
            "claim_query": state["claim_query"],
        }).content.strip()
        return {"search_query": query, "query_history": [query]}

    def retrieve_node(state):
        docs = state["vectorstore"].similarity_search(state["search_query"], k=6)
        return {"retrieved_docs": docs}

    def grade_relevance_node(state):
        grade = relevance_grader_chain.invoke({"claim_query": format_claim(state["claim_input"]),"search_query": state["search_query"],"retrieved_context": format_docs_for_grading(state["retrieved_docs"])})

        return {"relevance_grade": grade}

    def rewrite_query_node(state):
        new_query = rewrite_chain.invoke({
            "claim_query": state["claim_query"], "search_query": state["search_query"],
            "grade_reasoning": state["relevance_grade"].reasoning,
        }).content.strip()
        return {"search_query": new_query, "query_history": state["query_history"] + [new_query],
                "retry_count": state["retry_count"] + 1}

    def web_regulation_fallback_node(state):
        if not state["enable_web_fallback"] or state["tavily_client"] is None:
            return {"web_search_results": []}
        query = f"insurance regulation coverage requirements: {state['claim_query']}"
        try:
            response = state["tavily_client"].search(
                query=query, max_results=3, search_depth="basic",
                include_domains=AUTHORITATIVE_INSURANCE_DOMAINS,
            )
            results = [{"title": sanitize_text(r["title"]), "url": r["url"], "snippet": sanitize_text(r["content"][:300])}
                       for r in response.get("results", [])]
        except Exception:
            results = []
        return {"web_search_results": results}

    def decide_node(state):
        policy_context = format_docs_for_grading(
            state["retrieved_docs"]
        )

        regulatory_context = format_web_results(
            state.get("web_search_results", [])
        )

        grounding_feedback = ""

        if state.get("hallucination_grade"):
            grounding = state["hallucination_grade"]

            grounding_feedback = (
                "\n\nPREVIOUS DECISION CORRECTION:\n"
                f"Grounding reasoning: {grounding.reasoning}\n"
                f"Unsupported claims: {grounding.unsupported_claims}\n"
                f"Ignored conflicts: {grounding.ignored_conflicts}\n"
                "Correct the previous decision using only the retrieved evidence. "
                "Remove unsupported statements and invalid citations."
            )

        decision = decision_chain.invoke({"claim_query": format_claim(state["claim_input"]) + grounding_feedback,"retrieved_context": policy_context,"regulatory_context": regulatory_context})
        policy_context_lower = policy_context.lower()

        if decision.verdict == "approve":
            approval_terms = [
                "collision coverage",
                "comprehensive coverage",
                "full safety glass",
                "transportation expense",
                "rental",
                "medical payments",
                "uninsured motorist",
                "oem parts endorsement",
            ]

            if not any(
                term in policy_context_lower
                for term in approval_terms
            ):
                decision.verdict = "escalate"
                decision.reasoning += (
                    " Approval changed to escalation because no matching "
                    "coverage evidence was retrieved."
                )

        if decision.verdict == "deny":
            exclusion_terms = [
                "excluded",
                "no coverage",
                "not covered",
                "does not apply",
                "coverage is void",
                "coverage is suspended",
            ]

            if not any(
                term in policy_context_lower
                for term in exclusion_terms
            ):
                decision.verdict = "escalate"
                decision.reasoning += (
                    " Denial changed to escalation because no explicit "
                    "exclusion or limiting language was retrieved."
                )

        return {"decision": decision}

    def check_grounding_node(state):
        context = format_docs_for_grading(state["retrieved_docs"])
        grade = hallucination_chain.invoke({
            "retrieved_context": context, "verdict": state["decision"].verdict,
            "decision_reasoning": state["decision"].reasoning,
            "cited_clause_ids": state["decision"].cited_clause_ids,
        })
        return {"hallucination_grade": grade}

    def increment_decision_retry_node(state):
        return {"decision_retry_count": state["decision_retry_count"] + 1}

    def escalate_insufficient_evidence_node(state):
        reasoning = (f"No policy clauses met the relevance threshold ({state['relevance_threshold']}) "
                     f"after {state['retry_count']} retrieval attempt(s). "
                     f"Last grader reasoning: {state['relevance_grade'].reasoning}")
        if state["web_search_results"]:
            sources = "; ".join(f"{r['title']} ({r['url']})" for r in state["web_search_results"])
            reasoning += f"\n\nSupplementary regulatory research for the human reviewer: {sources}"
        else:
            reasoning += "\n\nNo supplementary regulatory sources found."
        return {
            "decision": Decision(
                verdict="escalate",
                confidence=0.95,
                reasoning=reasoning,
                cited_clause_ids=[],
                missing_information=[
                    "Relevant policy clauses or supporting claim evidence"
                ],
                recommended_action=(
                    "Send the claim to a human reviewer and obtain additional policy "
                    "or claim documentation."
                )
            )
        }
    def escalate_ungrounded_node(state):
        reasoning = (f"Automated decision ('{state['decision'].verdict}') failed grounding verification "
                     f"(confidence {state['hallucination_grade'].confidence:.2f}). Escalating for manual review. "
                     f"Original reasoning: {state['decision'].reasoning}")
        return {
            "decision": Decision(
                verdict="escalate",
                confidence=state["hallucination_grade"].confidence,
                reasoning=reasoning,
                cited_clause_ids=state["decision"].cited_clause_ids,
                missing_information=[
                    "Fully grounded evidence supporting the automated decision"
                ],
                recommended_action=(
                    "Send the claim to a human reviewer because the automated "
                    "decision failed grounding verification."
                )
            )
        }
    def finalize_node(state):
        requires_human_review = state["decision"].verdict == "escalate"
        priority = _escalation_priority(state.get("claim_input")) if requires_human_review else None
        record = AuditRecord(
            claim_id=state["claim_id"], claim_query=state["claim_query"],
            verdict=state["decision"].verdict, reasoning=state["decision"].reasoning,
            relevance_confidence=state["relevance_grade"].confidence if state["relevance_grade"] else None,
            hallucination_confidence=state["hallucination_grade"].confidence if state["hallucination_grade"] else None,
            cited_clause_ids=state["decision"].cited_clause_ids,
            retrieved_clause_ids=[d.metadata["clause_id"] for d in state["retrieved_docs"]],
            search_queries_tried=state["query_history"], retry_count=state["retry_count"],
            web_sources_consulted=state["web_search_results"],
            requires_human_review=requires_human_review,
            escalation_priority=priority,
            model_name_grader=GRADER_MODEL,
            model_name_decision=DECISION_MODEL,
            prompt_version=PROMPT_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(record.model_dump_json() + "\n")
        if requires_human_review:
            with open(HUMAN_REVIEW_QUEUE_PATH, "a") as f:
                f.write(record.model_dump_json() + "\n")
        return {"audit_record": record.model_dump()}

    # ----- Routing -----
    def route_after_validation(state):
        return "invalid_input" if state["validation_error"] else "build_query"

    def route_after_grade(state):
        grade = state["relevance_grade"]

        if (
            grade.relevant
            and grade.confidence >= state["relevance_threshold"]
        ):
            return "decide"

        if state["retry_count"] < state["max_retries"]:
            return "rewrite_query"

        return "web_fallback"
    def route_after_decision(state):
        if state["decision"].verdict == "escalate":
            return "finalize"

        return "check_grounding"
    def route_after_hallucination(state):
        decision = state["decision"]

    # An escalation is already the safe outcome.
    # Do not retry or escalate it again.
        if decision.verdict == "escalate":
            return "finalize"

        h = state["hallucination_grade"]

        if (
            h.grounded
            and h.citation_ids_valid
            and h.verdict_supported
            and h.confidence >= state["hallucination_threshold"]
        ):
            return "finalize"

        if state["decision_retry_count"] < MAX_DECISION_RETRIES:
            return "retry_decision"

        return "escalate_ungrounded"

    # ----- Graph assembly -----
    builder = StateGraph(ClaimState)
    for name, fn in [
        ("validate_claim", validate_claim_node),
        ("invalid_input", invalid_input_node),
        ("build_query", build_query_node),
        ("retrieve", retrieve_node),
        ("grade_relevance", grade_relevance_node),
        ("rewrite_query", rewrite_query_node),
        ("web_fallback", web_regulation_fallback_node),
        ("decide", decide_node),
        ("check_grounding", check_grounding_node),
        ("increment_decision_retry", increment_decision_retry_node),
        ("escalate_insufficient_evidence", escalate_insufficient_evidence_node),
        ("escalate_ungrounded", escalate_ungrounded_node),
        ("finalize", finalize_node),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "validate_claim")
    builder.add_conditional_edges("validate_claim", route_after_validation,
        {"build_query": "build_query", "invalid_input": "invalid_input"})
    builder.add_edge("invalid_input", "finalize")
    builder.add_edge("build_query", "retrieve")
    builder.add_edge("retrieve", "grade_relevance")
    builder.add_conditional_edges("grade_relevance", route_after_grade,
        {"rewrite_query": "rewrite_query", "decide": "decide", "web_fallback": "web_fallback"})
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("web_fallback", "escalate_insufficient_evidence")
    builder.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "finalize": "finalize",
            "check_grounding": "check_grounding",
        },
    )
    builder.add_conditional_edges("check_grounding", route_after_hallucination,
        {"finalize": "finalize", "retry_decision": "increment_decision_retry", "escalate_ungrounded": "escalate_ungrounded"})
    builder.add_edge("increment_decision_retry", "decide")
    builder.add_edge("escalate_insufficient_evidence", "finalize")
    builder.add_edge("escalate_ungrounded", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def adjudicate(graph, claim_id, claim_query, vectorstore, tavily_client,
               relevance_threshold=0.7, hallucination_threshold=0.7,
               max_retries=2, enable_web_fallback=True,
               policy_type=None, policy_number=None, claimant_name=None,
               date_of_loss=None, claimed_amount=None):

    max_retries = min(max_retries, MAX_RETRIEVAL_RETRIES_CAP)

    raw_claim_input = {
        "claim_query": claim_query,
        "policy_type": policy_type,
        "policy_number": policy_number,
        "claimant_name": claimant_name,
        "date_of_loss": date_of_loss,
        "claimed_amount": claimed_amount,
    }

    return graph.invoke({
        "claim_id": claim_id,
        "raw_claim_input": raw_claim_input,
        "claim_input": None,
        "validation_error": None,
        "claim_query": claim_query,
        "search_query": claim_query,
        "query_history": [],
        "vectorstore": vectorstore,
        "retrieved_docs": [], "relevance_grade": None,
        "retry_count": 0, "max_retries": max_retries, "relevance_threshold": relevance_threshold,
        "decision": None, "hallucination_grade": None, "decision_retry_count": 0,
        "hallucination_threshold": hallucination_threshold, "audit_record": None,
        "enable_web_fallback": enable_web_fallback, "tavily_client": tavily_client,
        "web_search_results": [],
        "verified_regulatory_results": [],

    })

from insurance_agent import build_vectorstore_from_files

test_files = [
    "PERSONAL AUTO INSURANCE POLICY.txt"
]

vectorstore = build_vectorstore_from_files(test_files)

results = vectorstore.similarity_search(
    "What collision damage is covered?",
    k=3
)

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Clause ID:", doc.metadata.get("clause_id"))
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:500])

import streamlit as st
import os, uuid
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



# ----------------- EMAIL NOTIFICATION FEATURE -----------------
def send_claim_email(record, receiver_email):
    """
    Sends the claim decision report to the email address
    entered by the user in the Streamlit interface.
    """
    sender_email = "pawasepramod@gmail.com"
    app_password = "rgct ynpt sduz ceng"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Claim Adjudication Result: {record['claim_id']} - {record['verdict'].upper()}"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    # Dynamic styling based on verdict
    color = "#28a745" if record['verdict'] == "approve" else "#dc3545" if record['verdict'] == "deny" else "#ffc107"

    # HTML Email Body with good fonts and layout
    html = f"""
    <html>
      <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; line-height: 1.6; padding: 20px; background-color: #f4f7f6;">
        <div style="max-width: 650px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <div style="background-color: #f8f9fa; padding: 25px; border-bottom: 2px solid {color};">
            <h2 style="margin: 0; color: #2c3e50; font-size: 24px;">🗂️ Claim Guard Notification</h2>
          </div>
          <div style="padding: 30px;">
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Claim ID:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee; text-align: right;">{record['claim_id']}</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Verdict:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee; text-align: right;">
                        <span style="color: {color}; font-weight: 800; font-size: 16px; text-transform: uppercase;">{record['verdict']}</span>
                    </td>
                </tr>
            </table>

            <h3 style="color: #2c3e50; margin-top: 30px; font-size: 18px;">📝 Claim Query</h3>
            <p style="background-color: #f1f3f5; padding: 15px; border-radius: 6px; font-size: 14px; border-left: 4px solid #ced4da;">
                {record['claim_query']}
            </p>

            <h3 style="color: #2c3e50; margin-top: 30px; font-size: 18px;">⚖️ AI Reasoning</h3>
            <p style="font-size: 14px; color: #555; background-color: #fffaf0; padding: 15px; border-radius: 6px; border: 1px solid #f0e6d2;">
                {record['reasoning']}
            </p>
          </div>
          <div style="background-color: #f8f9fa; padding: 15px 20px; text-align: center; font-size: 12px; color: #888; border-top: 1px solid #eee;">
            This is an automated message generated by the Claim Guard AI Agent. Please do not reply directly to this email.
          </div>
        </div>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    # Send Email (Mocked for safety since sample credentials are used)
    try:
        # To make this live, uncomment below and use a real SMTP server:
          server = smtplib.SMTP('smtp.gmail.com', 587)
          server.starttls()
          # Use the app_password here, NOT your regular gmail password
          server.login(sender_email, app_password)
          server.sendmail(sender_email, receiver_email, msg.as_string())
          server.quit()
          return True, f"Email successfully sent to {receiver_email}"
    except Exception as e:
        return False, str(e)


# ----------------- MAIN APP UI -----------------
st.set_page_config(page_title="Claims Adjudication Agent", page_icon="🗂️", layout="wide")
st.title("🗂️ Claim Guard AI Agent")
st.caption("Self-correcting retrieval, grounded decisions, and human escalation when evidence is weak.")

with st.sidebar:
    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key", type="password")
    tavily_key = st.text_input("Tavily API Key (optional)", type="password")

    st.header("⚙️ Settings")
    relevance_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.7, 0.05)
    hallucination_threshold = st.slider("Grounding threshold", 0.0, 1.0, 0.7, 0.05)
    max_retries = st.slider("Max retrieval retries", 0, 2, 2,
                             help="Hard-capped at 2 by the agent regardless of this setting.")
    enable_web_fallback = st.checkbox("Enable web regulation fallback", value=bool(tavily_key))

    st.header("📄 Policy Documents")
    uploaded_files = st.file_uploader("Upload policy docs (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    build_index = st.button("Build / Rebuild Index")

    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "rb") as f:
            st.download_button("⬇️ Download audit log (JSON)", f, file_name="audit_log.jsonl")
    if os.path.exists(HUMAN_REVIEW_QUEUE_PATH):
        with open(HUMAN_REVIEW_QUEUE_PATH, "rb") as f:
            st.download_button("⬇️ Download human review queue (JSON)", f, file_name="human_review_queue.jsonl")

# Initialize session states securely
for key, default in [("vectorstore", None), ("graph", None), ("history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

if build_index:
    if not uploaded_files:
        st.sidebar.error("Upload at least one document first.")
    else:
        os.makedirs("uploaded_docs", exist_ok=True)
        paths = []
        for f in uploaded_files:
            path = os.path.join("uploaded_docs", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)
        with st.spinner("Chunking and embedding documents (Cleaning ASCII data)..."):
            try:
                st.session_state.vectorstore = build_vectorstore_from_files(paths)
                st.sidebar.success(f"Indexed {len(paths)} document(s).")
            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")

if groq_key and st.session_state.graph is None:
    st.session_state.graph = build_claims_graph(groq_key)

tavily_client = None
if tavily_key:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=tavily_key)

st.subheader("Submit a claim")


claim_query = st.text_area(
    "Describe the claim scenario",
    height=120,
    placeholder=(
        "Describe how the vehicle was damaged, who was driving, "
        "when it happened, and what coverage is being requested."
    )
)

with st.expander("➕ Additional claim details (optional)"):
    col1, col2 = st.columns(2)
    with col1:
        policy_type = "auto"
        st.info("Supported policy type: Personal Auto Insurance")
        policy_number = st.text_input("Policy number")
        claimant_name = st.text_input("Claimant name")
    with col2:
        date_of_loss = st.text_input("Date of loss")
        claimed_amount = st.number_input("Claimed amount ($)", min_value=0.0, value=0.0, step=100.0)

send_email_notification = st.checkbox("📧 Send Email Notification on Decision", value=True)
recipient_email = ""

if send_email_notification:
    recipient_email = st.text_input(
        "Recipient Email Address",
        placeholder="example@gmail.com",
        help="The claim decision report will be sent to this email address."
    )
submit = st.button("Adjudicate Claim", type="primary")

if submit:
    if not groq_key:
        st.error("Enter your Groq API key in the sidebar.")
    elif not claim_query.strip():
        st.error("Describe a claim scenario first.")
    elif st.session_state.vectorstore is None:
        st.error("Upload and index at least one policy document in the sidebar first.")
    elif send_email_notification and not recipient_email.strip():
        st.error("Enter the recipient email address.")

    elif (
        send_email_notification
        and (
            "@" not in recipient_email
            or "." not in recipient_email.split("@")[-1]
        )
    ):
        st.error("Enter a valid recipient email address.")
    else:
        claim_id = str(uuid.uuid4())[:8]
        with st.spinner("Validating, retrieving evidence, grading, and adjudicating..."):
            try:
                result = adjudicate(
                    st.session_state.graph, claim_id, claim_query,
                    st.session_state.vectorstore, tavily_client,
                    relevance_threshold, hallucination_threshold,
                    max_retries, enable_web_fallback,
                    policy_type=policy_type or None,
                    policy_number=policy_number or None,
                    claimant_name=claimant_name or None,
                    date_of_loss=date_of_loss or None,
                    claimed_amount=claimed_amount if claimed_amount > 0 else None,
                )

                decision = result["decision"]

                color = {
                    "approve": "green",
                    "deny": "red",
                    "escalate": "orange"
                }[decision.verdict]

                st.markdown(
                    f"### Verdict: :{color}[{decision.verdict.upper()}]"
                )

                st.metric(
                    "Decision Confidence",
                    f"{decision.confidence * 100:.0f}%"
                )

                st.subheader("Reasoning")
                st.write(decision.reasoning)

                st.subheader("Recommended Action")
                st.info(decision.recommended_action)

                if decision.missing_information:
                    st.subheader("Missing Information")

                    for item in decision.missing_information:
                        st.warning(item)

                if decision.cited_clause_ids:
                    st.subheader("Supporting Clause IDs")

                    for clause_id in decision.cited_clause_ids:
                        st.code(clause_id)

                if result["audit_record"].get("requires_human_review"):
                    st.warning(
                        f"🧑‍⚖️ Routed to human adjuster review "
                        f"(priority: {result['audit_record'].get('escalation_priority')})."
                    )
                st.subheader("Agent Workflow")

                query_history = result["audit_record"].get("search_queries_tried", [])
                retry_count = result["audit_record"].get("retry_count", 0)
                relevance_score = result["audit_record"].get("relevance_confidence")
                grounding_score = result["audit_record"].get("hallucination_confidence")
                human_review = result["audit_record"].get("requires_human_review", False)

                st.write("1. Policy documents retrieved from Chroma")

                if relevance_score is not None:
                    st.write(f"2. Retrieval relevance graded: {relevance_score:.2f}")
                else:
                    st.write("2. Retrieval relevance graded")

                if retry_count > 0:
                    st.write(f"3. Query rewritten and retrieval retried {retry_count} time(s)")
                else:
                    st.write("3. No query rewrite was required")

                st.write(f"4. Decision generated: {decision.verdict.upper()}")

                if grounding_score is not None:
                    st.write(f"5. Grounding verified with confidence: {grounding_score:.2f}")
                else:
                    st.write("5. Grounding verification completed")

                if human_review:
                    st.write("6. Claim routed to human adjuster")
                else:
                    st.write("6. Automated adjudication completed")

                if query_history:
                    with st.expander("Queries attempted"):
                        for index, query in enumerate(query_history, start=1):
                            st.write(f"{index}. {query}")
                with st.expander("Full audit record"):
                    st.json(result["audit_record"])

                st.session_state.history.append(result["audit_record"])

                if send_email_notification:
                    success, msg = send_claim_email(
                        result["audit_record"],
                        recipient_email.strip()
                    )

                    if success:
                        st.toast(f"✅ {msg}", icon="📧")
                    else:
                        st.toast(
                            f"❌ Failed to send email: {msg}",
                            icon="🚨"
                        )

            except Exception as e:
                st.error("⚠️ The agent hit an error and could not complete automated adjudication.")
                st.write(f"Details: {e}")
                st.warning(f"Recommendation: escalate claim `{claim_id}` to a human reviewer manually.")

# ----------------- SESSION HISTORY & CSV EXPORT -----------------
if st.session_state.history:
    st.subheader("📋 Session claim history")

    # Map the session history to a Pandas DataFrame for full data visibility
    df_history = pd.DataFrame([
        {
            "Claim ID": r["claim_id"],
            "Verdict": r["verdict"].upper(),
            "Human Review Needed": r.get("requires_human_review", False),
            "Claim Query": r["claim_query"], # Full query instead of truncated
            "Reasoning": r["reasoning"]      # New: Added Full reasoning
        }
        for r in reversed(st.session_state.history) # Latest first
    ])

    # Display the full dataframe in Streamlit
    st.dataframe(df_history, use_container_width=True)

    # Add a clean CSV download button
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv_data,
        file_name="session_claim_history.csv",
        mime="text/csv",
        type="secondary"
    )	
	
	
