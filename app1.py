
import os
import uuid
from datetime import datetime, timezone
from typing import TypedDict, List, Optional, Literal

import streamlit as st
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

AUDIT_LOG_PATH = "audit_log.jsonl"
AUTHORITATIVE_INSURANCE_DOMAINS = ["naic.org", "content.naic.org"]

# ======================================================================
# Schemas
# ======================================================================
class RelevanceGrade(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Decision(BaseModel):
    verdict: Literal["approve", "deny", "escalate"]
    reasoning: str
    cited_clause_ids: List[str]


class HallucinationGrade(BaseModel):
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
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
    timestamp: str


class ClaimState(TypedDict):
    claim_id: str
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


# ======================================================================
# System prompts
# ======================================================================
RELEVANCE_GRADER_SYSTEM_PROMPT = """You are a meticulous insurance claims analyst evaluating whether retrieved policy text actually addresses a specific claim scenario.

Score how well the retrieved clauses cover the claim scenario on a 0.0-1.0 scale:
- 1.0: The clauses directly and specifically resolve the claim scenario (a coverage, exclusion, or condition that clearly applies)
- 0.5: The clauses are topically related but don't clearly resolve the scenario
- 0.0: The clauses are unrelated to the claim scenario

Be strict. Surface-level topical overlap is not enough for a high score if the specific circumstances differ. Base your score only on the retrieved text below - do not use outside insurance knowledge.

Claim scenario: {claim_query}

Retrieved policy clauses:
{retrieved_context}"""

REWRITE_SYSTEM_PROMPT = """You are refining a search query against an insurance policy database.
The previous search attempt did not retrieve clauses that clearly resolve the claim scenario.

Original claim scenario: {claim_query}
Previous search query: {search_query}
Why the previous retrieval was weak: {grade_reasoning}

Write ONE improved search query, phrased using terms likely to appear in policy documents
(coverage, exclusion, section names) rather than conversational language.
Return ONLY the query text, nothing else."""

DECISION_SYSTEM_PROMPT = """You are a senior insurance claims adjudicator. Decide this claim using ONLY the retrieved policy clauses below � never use outside knowledge about insurance norms.

Claim scenario: {claim_query}

Retrieved policy clauses:
{retrieved_context}

Rules:
- APPROVE only if a clause explicitly supports coverage for this exact scenario.
- DENY only if a clause explicitly excludes or fails to cover this scenario.
- ESCALATE if the clauses are ambiguous, conflicting, or don't clearly resolve it either way.
- Every factual statement in your reasoning must trace to a specific clause ID. Never invent terms, dollar amounts, or conditions not present in the retrieved text.
- List the clause_id(s) you relied on.
Be conservative: when in doubt, escalate rather than guess."""

HALLUCINATION_GRADER_SYSTEM_PROMPT = """You are a fact-checking auditor. Verify a claims decision is fully grounded in the retrieved policy text � not inferred or invented.

Retrieved policy clauses:
{retrieved_context}

Decision made: {verdict}
Decision reasoning: {decision_reasoning}
Clauses cited: {cited_clause_ids}

Check whether every factual statement (dollar amounts, conditions, exclusions) actually appears in the retrieved clauses. Flag anything fabricated or extrapolated. Score 'grounded' true only if there is no fabrication."""


# ======================================================================
# Ingestion helpers
# ======================================================================
def build_vectorstore_from_files(file_paths: List[str]) -> FAISS:
    raw_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path)
        raw_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        chunk.metadata["clause_id"] = f"{source}::chunk_{i}"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def format_docs_for_grading(docs):
    return "\n\n".join(f"[{d.metadata['clause_id']}]\n{d.page_content}" for d in docs)


# ======================================================================
# Graph builder
# ======================================================================
def build_claims_graph(groq_api_key: str):
    grader_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=groq_api_key)
    strong_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=groq_api_key)

    relevance_grader_chain = (
        ChatPromptTemplate.from_messages([("system", RELEVANCE_GRADER_SYSTEM_PROMPT)])
        | grader_llm.with_structured_output(RelevanceGrade)
    )
    rewrite_chain = ChatPromptTemplate.from_messages([("system", REWRITE_SYSTEM_PROMPT)]) | grader_llm
    decision_chain = (
        ChatPromptTemplate.from_messages([("system", DECISION_SYSTEM_PROMPT)])
        | strong_llm.with_structured_output(Decision)
    )
    hallucination_chain = (
        ChatPromptTemplate.from_messages([("system", HALLUCINATION_GRADER_SYSTEM_PROMPT)])
        | grader_llm.with_structured_output(HallucinationGrade)
    )

    def retrieve_node(state):
        docs = state["vectorstore"].similarity_search(state["search_query"], k=3)
        return {"retrieved_docs": docs}

    def grade_node(state):
        context = format_docs_for_grading(state["retrieved_docs"])
        grade = relevance_grader_chain.invoke({"claim_query": state["claim_query"], "retrieved_context": context})
        return {"relevance_grade": grade}

    def rewrite_node(state):
        new_query = rewrite_chain.invoke({
            "claim_query": state["claim_query"], "search_query": state["search_query"],
            "grade_reasoning": state["relevance_grade"].reasoning,
        }).content.strip()
        return {
            "search_query": new_query,
            "query_history": state["query_history"] + [new_query],
            "retry_count": state["retry_count"] + 1,
        }

    def web_regulation_fallback_node(state):
        if not state["enable_web_fallback"] or state["tavily_client"] is None:
            return {"web_search_results": []}
        query = f"insurance regulation coverage requirements: {state['claim_query']}"
        try:
            response = state["tavily_client"].search(
                query=query, max_results=3, search_depth="basic",
                include_domains=AUTHORITATIVE_INSURANCE_DOMAINS,
            )
            results = [{"title": r["title"], "url": r["url"], "snippet": r["content"][:300]}
                       for r in response.get("results", [])]
        except Exception:
            results = []
        return {"web_search_results": results}

    def decide_node(state):
        context = format_docs_for_grading(state["retrieved_docs"])
        decision = decision_chain.invoke({"claim_query": state["claim_query"], "retrieved_context": context})
        return {"decision": decision}

    def check_hallucination_node(state):
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
        return {"decision": Decision(verdict="escalate", reasoning=reasoning, cited_clause_ids=[])}

    def escalate_ungrounded_node(state):
        reasoning = (f"Automated decision ('{state['decision'].verdict}') failed grounding verification "
                     f"(confidence {state['hallucination_grade'].confidence:.2f}). Escalating for manual review. "
                     f"Original reasoning: {state['decision'].reasoning}")
        return {"decision": Decision(verdict="escalate", reasoning=reasoning,
                                      cited_clause_ids=state["decision"].cited_clause_ids)}

    def finalize_node(state):
        record = AuditRecord(
            claim_id=state["claim_id"], claim_query=state["claim_query"],
            verdict=state["decision"].verdict, reasoning=state["decision"].reasoning,
            relevance_confidence=state["relevance_grade"].confidence if state["relevance_grade"] else None,
            hallucination_confidence=state["hallucination_grade"].confidence if state["hallucination_grade"] else None,
            cited_clause_ids=state["decision"].cited_clause_ids,
            retrieved_clause_ids=[d.metadata["clause_id"] for d in state["retrieved_docs"]],
            search_queries_tried=state["query_history"], retry_count=state["retry_count"],
            web_sources_consulted=state["web_search_results"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(record.model_dump_json() + "\n")
        return {"audit_record": record.model_dump()}

    def route_after_grade(state):
        grade = state["relevance_grade"]
        if grade.confidence >= state["relevance_threshold"]:
            return "decide"
        if state["retry_count"] < state["max_retries"]:
            return "rewrite"
        return "web_fallback"

    def route_after_hallucination(state):
        h = state["hallucination_grade"]
        if h.grounded and h.confidence >= state["hallucination_threshold"]:
            return "finalize"
        if state["decision_retry_count"] < 1:
            return "retry_decision"
        return "escalate_ungrounded"

    builder = StateGraph(ClaimState)
    for name, fn in [
        ("retrieve", retrieve_node), ("grade", grade_node), ("rewrite", rewrite_node),
        ("web_fallback", web_regulation_fallback_node), ("decide", decide_node),
        ("check_hallucination", check_hallucination_node),
        ("increment_decision_retry", increment_decision_retry_node),
        ("escalate_insufficient_evidence", escalate_insufficient_evidence_node),
        ("escalate_ungrounded", escalate_ungrounded_node), ("finalize", finalize_node),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade,
        {"rewrite": "rewrite", "decide": "decide", "web_fallback": "web_fallback"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("web_fallback", "escalate_insufficient_evidence")
    builder.add_edge("decide", "check_hallucination")
    builder.add_conditional_edges("check_hallucination", route_after_hallucination,
        {"finalize": "finalize", "retry_decision": "increment_decision_retry", "escalate_ungrounded": "escalate_ungrounded"})
    builder.add_edge("increment_decision_retry", "decide")
    builder.add_edge("escalate_insufficient_evidence", "finalize")
    builder.add_edge("escalate_ungrounded", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def adjudicate(graph, claim_id, claim_query, vectorstore, tavily_client,
               relevance_threshold=0.7, hallucination_threshold=0.7,
               max_retries=2, enable_web_fallback=True):
    return graph.invoke({
        "claim_id": claim_id, "claim_query": claim_query, "search_query": claim_query,
        "query_history": [claim_query], "vectorstore": vectorstore,
        "retrieved_docs": [], "relevance_grade": None,
        "retry_count": 0, "max_retries": max_retries, "relevance_threshold": relevance_threshold,
        "decision": None, "hallucination_grade": None, "decision_retry_count": 0,
        "hallucination_threshold": hallucination_threshold, "audit_record": None,
        "enable_web_fallback": enable_web_fallback, "tavily_client": tavily_client,
        "web_search_results": [],
    })


# ======================================================================
# Streamlit UI
# ======================================================================
st.set_page_config(page_title="Claims Adjudication Agent", page_icon="???", layout="wide")
st.title("??? Insurance Claims Adjudication Agent")
st.caption("Self-correcting retrieval, grounded decisions, and human escalation when evidence is weak.")

with st.sidebar:
    st.header("?? API Keys")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="Enter Groq API key")
    tavily_key = st.text_input("Tavily API Key (optional)", type="password", placeholder="Enter Tavily API key")

    st.header("?? Settings")
    relevance_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.7, 0.05)
    hallucination_threshold = st.slider("Grounding threshold", 0.0, 1.0, 0.7, 0.05)
    max_retries = st.slider("Max retrieval retries", 0, 3, 2)
    enable_web_fallback = st.checkbox("Enable web regulation fallback", value=bool(tavily_key))

    st.header("?? Policy Documents")
    uploaded_files = st.file_uploader("Upload policy docs (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    build_index = st.button("Build / Rebuild Index")

    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "rb") as f:
            st.download_button("?? Download audit log", f, file_name="audit_log.jsonl")

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
        with st.spinner("Chunking and embedding documents..."):
            try:
                st.session_state.vectorstore = build_vectorstore_from_files(paths)
                st.sidebar.success(f"Indexed {len(paths)} document(s).")
            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")

if groq_key and st.session_state.graph is None:
    st.session_state.graph = build_claims_graph(groq_key)

tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

st.subheader("Submit a claim")
claim_query = st.text_area(
    "Describe the claim scenario", height=100,
    placeholder="e.g. Customer's basement flooded because a pipe burst under the sink..."
)
submit = st.button("Adjudicate Claim", type="primary")

if submit:
    if not groq_key:
        st.error("Enter your Groq API key in the sidebar.")
    elif not claim_query.strip():
        st.error("Describe a claim scenario first.")
    elif st.session_state.vectorstore is None:
        st.error("Upload and index at least one policy document in the sidebar first.")
    else:
        claim_id = str(uuid.uuid4())[:8]
        with st.spinner("Retrieving evidence, grading, and adjudicating..."):
            try:
                result = adjudicate(
                    st.session_state.graph, claim_id, claim_query,
                    st.session_state.vectorstore, tavily_client,
                    relevance_threshold, hallucination_threshold,
                    max_retries, enable_web_fallback,
                )
                decision = result["decision"]
                color = {"approve": "green", "deny": "red", "escalate": "orange"}[decision.verdict]
                st.markdown(f"### Verdict: :{color}[{decision.verdict.upper()}]")
                st.write(decision.reasoning)
                if decision.cited_clause_ids:
                    st.caption("Cited clauses: " + ", ".join(decision.cited_clause_ids))
                with st.expander("Full audit record"):
                    st.json(result["audit_record"])
                st.session_state.history.append(result["audit_record"])
            except Exception as e:
                st.error("?? The agent hit an error and could not complete automated adjudication.")
                st.write(f"Details: {e}")
                st.warning(f"Recommendation: escalate claim `{claim_id}` to a human reviewer manually.")

if st.session_state.history:
    st.subheader("?? Session claim history")
    st.dataframe(
        [{"claim_id": r["claim_id"], "verdict": r["verdict"], "claim_query": r["claim_query"][:60]}
         for r in st.session_state.history],
        use_container_width=True,
    )
