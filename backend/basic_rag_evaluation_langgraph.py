# basic_rag_evaluation_langgraph.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

# ---- LangSmith / LangChain tracing env ----
import os
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "Sentri-GenAI-Evals")

# ---- traceable decorator (safe fallback) ----
try:
    from langsmith import traceable
except Exception:
    try:
        from langsmith.run_helpers import traceable
    except Exception:
        def traceable(*_a, **_k):
            def _wrap(f): return f
            return _wrap
import re
import atexit
import json
from typing import Any, Dict, List, Tuple, TypedDict
from pathlib import Path
import weaviate
from langsmith import Client
from langsmith.evaluation import evaluate
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryByteStore
from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import ChatOllama

# ------------- Config -------------
DATA_DIR = Path(__file__).parent / "data"
KB_PATH = DATA_DIR / "kb.json"
DATASET_NAME = "SecurityQA-Questions"
QUESTIONS_PATH = DATA_DIR / "questions.json"
GROUNDTRUTH_PATH = DATA_DIR / "GroundTruth.json"  # <-- ADDED: use GT for judging

CLASS_PREFIX = "SecurityQA"
GLOBAL_INDEX = f"{CLASS_PREFIX}_GLOBAL"      # one collection for all questions
TEXT_KEY = "question_text"                   # field to store question text in Weaviate
ID_KEY = "doc_id"                            # links child (question) -> parent (answer)

# Adaptive loop guards
MAX_RETRIEVAL_LOOPS = 2     # how many times we'll rewrite queries and re-retrieve
MAX_REGEN_LOOPS = 3         # how many times we'll regenerate the answer if hallucinating/low quality

EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"

TOP_K=4
PER_QUERY_K = 4  # how many parents to retrieve for generation
RRF = 60 #RRF constant
NUM_QUERY_VARIANTS = 5 # number of multiple queries to generate
# --- lazy globals ---
_client_wv: weaviate.WeaviateClient | None = None
_global_retriever: MultiVectorRetriever | None = None
_APP = None  # LangGraph app


# ------------- KB helpers -------------
def read_kb(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for it in data:
        out.append({
            "id": it.get("id"),
            "category": it.get("category"),
            "question": it.get("question"),
            "answer": it.get("answer"),
            "created_at": it.get("created_at"),
            "updated_at": it.get("updated_at"),
        })
    return out


def build_child_docs(items: List[Dict[str, Any]]) -> List[Document]:
    # Child = Question (embedded & indexed)
    docs: List[Document] = []
    for it in items:
        docs.append(Document(
            page_content=it["question"],
            metadata={
                ID_KEY: str(it["id"]),
                "category": it.get("category"),
                "created_at": it.get("created_at"),
                "updated_at": it.get("updated_at"),
                "question": it.get("question"),
            },
        ))
    return docs


def build_parent_docs(items: List[Dict[str, Any]]) -> Tuple[List[str], List[Document]]:
    # Parent = Answer (stored in byte_store, fetched by ID_KEY)
    ids: List[str] = []
    parents: List[Document] = []
    for it in items:
        pid = str(it["id"])
        ids.append(pid)
        parents.append(Document(
            page_content=it["answer"],
            metadata={
                "id": pid,
                "category": it.get("category"),
                "question": it.get("question"),
                "created_at": it.get("created_at"),
                "updated_at": it.get("updated_at"),
            },
        ))
    return ids, parents


# ------------- Build single global retriever -------------
def connect_weaviate_local():
    return weaviate.connect_to_local()

def build_global_retriever(kb_items: List[Dict[str, Any]], client) -> MultiVectorRetriever:
    # Build a single Weaviate index with ALL questions as children
    child_docs = build_child_docs(kb_items)
    vs = WeaviateVectorStore.from_documents(
        documents=child_docs,
        embedding=OpenAIEmbeddings(model=EMBED_MODEL),
        client=client,
        index_name=GLOBAL_INDEX,
        text_key=TEXT_KEY,
    )
    retriever = MultiVectorRetriever(
        vectorstore=vs,
        byte_store=InMemoryByteStore(),
        id_key=ID_KEY,
    )
    ids, parents = build_parent_docs(kb_items)
    retriever.docstore.mset(list(zip(ids, parents)))
    return retriever


# ------------- Generation -------------
GEN_LLM = ChatOpenAI(
    model=GEN_MODEL,
    temperature=0,
)

GEN_SYSTEM = (
    "You help complete vendor security questionnaires. Answer ONLY from the provided context. "
    "If the answer is not present, reply exactly: 'No information available in the knowledge base.' "
    "Keep answers concise (2–6 sentences)."
)

_CIT_PATS = [
    re.compile(r"\[\d+(?:\s*(?:,|-)\s*\d+)*\]"),   # [1], [1,2], [1-3]
    re.compile(r"【\d+†[^】]*】"),                  # -style
    re.compile(r"\(see\s*\[\d+[^\]]*\]\)", re.I),  # (see [1]) etc.
]

def _strip_citations(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _CIT_PATS:
        out = pat.sub("", out)
    # Remove trailing “Sources:” sections if any slipped in
    out = re.sub(r"(?is)\n?\s*Sources?:.*$", "", out).strip()
    return out

def format_ctx(objs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(objs, 1):
        m = d.metadata or {}
        lines.append(f"[{i}] {m.get('category','doc')}\nQ: {m.get('question')}\nA: {d.page_content}\n")
    return "\n".join(lines) if lines else "(no context)"

def generate_answer(question: str, parents: List[Document]) -> str:
    if not parents:
        return "No information available in the knowledge base."
    system = {"role": "system", "content": GEN_SYSTEM}
    user = {
        "role": "user",
        "content": f"Question: {question}\n\nContext:\n{format_ctx(parents)}\n\nAnswer using only the Context."
    }
    resp = GEN_LLM.invoke([system, user])
    ans = getattr(resp, "content", str(resp)).strip()
    return _strip_citations(ans)

class QueryVariants(BaseModel):
    queries: list[str] = Field(
        ..., 
        min_length=NUM_QUERY_VARIANTS, 
        max_length=NUM_QUERY_VARIANTS,
        description=f"Exactly {NUM_QUERY_VARIANTS} query variants."
    )

EXPAND_LLM = ChatOpenAI(model=GEN_MODEL, temperature=0)
template = """You are a helpful assistant that generates multiple search queries based on a single input query.
Rewrite the QUESTION into {n} concise, distinct search queries that preserve meaning but vary wording.
Avoid filler, no answers, no quotes, no numbering or bullets. One query per line. 

Question: {question}
OUTPUT:"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion
    | EXPAND_LLM
    | StrOutputParser()
)

def _clean_split_lines(text:str) -> list[str]:
    lines = []
    for ln in str(text).splitlines():
        ln = re.sub(r'^\s*(?:[-*]|\d+[.)\:])\s*', '', ln).strip()
        if ln:
            lines.append(ln)
    return lines

def expand_queries(question:str) -> list[str]:
    raw = generate_queries.invoke({"question": question, "n": NUM_QUERY_VARIANTS})
    lines = _clean_split_lines(raw)
    #dedup
    out, seen = [], set()
    for q in lines:
        k=q.lower()
        if k not in seen:
            seen.add(k)
            out.append(q)
    
    if len(out) < NUM_QUERY_VARIANTS:
        out += [question] * (NUM_QUERY_VARIANTS - len(out))
    
    return out[:NUM_QUERY_VARIANTS]

def rrf_fuse(lists: list[list[Document]], rrf_k: int = RRF, top_k: int =TOP_K)-> list[Document]:
    scores, keep = {}, {}
    for L in lists:
        for rank, d in enumerate(L, 1):
            did = d.metadata.get("id") if d and d.metadata else None
            if not did:
                continue
            scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
            keep.setdefault(did, d)
    ranked = sorted(scores.items(), key = lambda kv:kv[1], reverse=True)
    return [keep[did] for did, _ in ranked[:top_k]]

# ---------------- Runtime graders for adaptive loop ----------------
class RtRetrievalGrade(BaseModel):
    explanation: str
    useful: bool  # True if the retrieved docs are relevant/helpful to answer the question

class RtGroundednessGrade(BaseModel):
    explanation: str
    grounded: bool  # True if answer is fully supported by the retrieved facts

class RtAnswerQualityGrade(BaseModel):
    explanation: str
    useful: bool  # True if the answer is concise and directly addresses the question

_RT_JUDGE = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

_RT_RETRIEVAL_SYSTEM = (
    "You judge whether the supplied FACTS are relevant to answering the QUESTION. "
    "Mark True if at least some FACTS are clearly helpful; False if unrelated."
)
_rt_retrieval_llm = _RT_JUDGE.with_structured_output(RtRetrievalGrade)

def rt_grade_retrieval(question: str, docs: List[Document]) -> RtRetrievalGrade:
    facts = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    prompt = f"QUESTION:\n{question}\n\nFACTS:\n{facts}"
    return _rt_retrieval_llm.invoke(
        [{"role": "system", "content": _RT_RETRIEVAL_SYSTEM},
         {"role": "user", "content": prompt}]
    )

_RT_GROUNDED_SYSTEM = (
    "You judge whether the STUDENT ANSWER is fully supported by the FACTS. "
    "If any material claim is not supported, mark grounded=False."
)
_rt_grounded_llm = _RT_JUDGE.with_structured_output(RtGroundednessGrade)

def rt_grade_groundedness(answer: str, docs: List[Document]) -> RtGroundednessGrade:
    facts = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    prompt = f"FACTS:\n{facts}\n\nSTUDENT ANSWER:\n{answer}"
    return _rt_grounded_llm.invoke(
        [{"role": "system", "content": _RT_GROUNDED_SYSTEM},
         {"role": "user", "content": prompt}]
    )

_RT_ANSWER_QUALITY_SYSTEM = (
    "You judge whether the STUDENT ANSWER is concise and directly addresses the QUESTION."
)
_rt_answer_quality_llm = _RT_JUDGE.with_structured_output(RtAnswerQualityGrade)

def rt_grade_answer_quality(question: str, answer: str) -> RtAnswerQualityGrade:
    prompt = f"QUESTION:\n{question}\n\nSTUDENT ANSWER:\n{answer}"
    return _rt_answer_quality_llm.invoke(
        [{"role": "system", "content": _RT_ANSWER_QUALITY_SYSTEM},
         {"role": "user", "content": prompt}]
    )
# -------------------------------------------------------------------

# ------------- LangGraph state + app -------------
class RAGState(TypedDict):
    question: str
    queries: list[str]
    parents: list[Document]
    answer: str
    # runtime grading & control
    retrieval_ok: bool
    grounded: bool
    answer_ok: bool
    retries_retrieval: int
    retries_generation: int

def _retrieve_topk(query: str, k: int) -> List[Document]:
    # MultiVectorRetriever.invoke may or may not accept n_results; handle both.
    try:
        return _global_retriever.invoke(query, n_results=k)  # type: ignore[arg-type]
    except TypeError:
        docs = _global_retriever.invoke(query)
        return docs[:k]

def make_app():
    g = StateGraph(RAGState)

    # ---------------- NODES ----------------

    @traceable(name="expand_node")
    def expand_node(state: RAGState) -> RAGState:
        variants = expand_queries(state["question"])
        # include original + rewrites (dedup)
        all_qs = [state["question"], *variants]
        uniq = []
        seen = set()
        for q in all_qs:
            k = q.strip().lower()
            if k and k not in seen:
                seen.add(k)
                uniq.append(q.strip())
        return {**state, "queries": uniq[: 1 + NUM_QUERY_VARIANTS]}

    @traceable(name="retrieve_node")
    def retrieve_node(state: RAGState) -> RAGState:
        qlist = state.get("queries") or [state["question"]]
        per_lists: list[list[Document]] = []
        with ThreadPoolExecutor(max_workers=min(8, len(qlist))) as ex:
            futs = {ex.submit(_retrieve_topk, q, PER_QUERY_K): q for q in qlist}
            for fut in as_completed(futs):
                try:
                    per_lists.append(fut.result() or [])
                except Exception:
                    per_lists.append([])
        fused = rrf_fuse(per_lists, rrf_k=RRF, top_k=TOP_K)
        return {**state, "parents": fused}

    @traceable(name="grade_retrieval")
    def grade_retrieval_node(state: RAGState) -> RAGState:
        grade = rt_grade_retrieval(state["question"], state.get("parents", []))
        return {**state, "retrieval_ok": bool(grade.useful)}

    @traceable(name="rewrite_queries")
    def rewrite_queries_node(state: RAGState) -> RAGState:
        # Try different phrasings. Include already-used queries to discourage repeats.
        tried = set([q.lower() for q in (state.get("queries") or [])])
        base = state["question"]
        rewrites = expand_queries(base)
        # de-dup against what we already tried
        new_qs = [q for q in rewrites if q.lower() not in tried] or rewrites
        return {
            **state,
            "queries": [base, *new_qs][: 1 + NUM_QUERY_VARIANTS],
            "retries_retrieval": state.get("retries_retrieval", 0) + 1,
        }

    @traceable(name="generate_node")
    def generate_node(state: RAGState) -> RAGState:
        ans = generate_answer(state["question"], state.get("parents", []))
        return {**state, "answer": ans}

    @traceable(name="grade_answer")
    def grade_answer_node(state: RAGState) -> RAGState:
        grounded_grade = rt_grade_groundedness(state.get("answer", ""), state.get("parents", []))
        quality_grade = rt_grade_answer_quality(state["question"], state.get("answer", ""))
        return {
            **state,
            "grounded": bool(grounded_grade.grounded),
            "answer_ok": bool(quality_grade.useful),
        }

    @traceable(name="bump_regen")
    def bump_regen_node(state: RAGState) -> RAGState:
        # Only used when we decide to regenerate the answer
        return {**state, "retries_generation": state.get("retries_generation", 0) + 1}

    # ---------------- ROUTERS ----------------

    def _route_after_retrieval(state: RAGState) -> str:
        """
        Decide whether to retry retrieval (rewrite queries) or move to generate.
        """
        bad = not state.get("retrieval_ok", True)
        too_many = state.get("retries_retrieval", 0) >= MAX_RETRIEVAL_LOOPS
        return "retry" if (bad and not too_many) else "to_gen"

    def _route_after_answer(state: RAGState) -> str:
        """
        Decide what to do after grading the answer.

        - If the answer does NOT answer the question (answer_ok = False)
          and we still have retrieval retries left -> go back to retrieval loop
          via 'rewrite_queries'.

        - Else if the answer DOES answer the question but is ungrounded
          and we still have regen budget -> regenerate with same docs.

        - Else -> finish.
        """
        answer_ok = state.get("answer_ok", True)
        grounded = state.get("grounded", True)
        ret_tries = state.get("retries_retrieval", 0)
        gen_tries = state.get("retries_generation", 0)

        # 1) Answer doesn't really answer the question -> treat as retrieval failure
        if (not answer_ok) and ret_tries < MAX_RETRIEVAL_LOOPS:
            return "to_retrieval"

        # 2) Answer is on-topic but not grounded -> treat as generation failure
        if (not grounded) and gen_tries < MAX_REGEN_LOOPS:
            return "to_regen"

        # 3) Otherwise we're done
        return "to_end"

    # ---------------- WIRING ----------------

    g.add_node("expand", expand_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade_retrieval", grade_retrieval_node)
    g.add_node("rewrite_queries", rewrite_queries_node)
    g.add_node("generate", generate_node)
    g.add_node("grade_answer", grade_answer_node)
    g.add_node("bump_regen", bump_regen_node)

    g.set_entry_point("expand")

    # Straight edges
    g.add_edge("expand", "retrieve")
    g.add_edge("retrieve", "grade_retrieval")
    g.add_edge("rewrite_queries", "retrieve")
    g.add_edge("generate", "grade_answer")
    g.add_edge("bump_regen", "generate")

    # Conditional edges after retrieval grading
    g.add_conditional_edges(
        "grade_retrieval",
        _route_after_retrieval,
        {
            "retry": "rewrite_queries",  # rewrite + re-retrieve
            "to_gen": "generate",        # go ahead and answer
        },
    )

    # Conditional edges after answer grading
    g.add_conditional_edges(
        "grade_answer",
        _route_after_answer,
        {
            "to_retrieval": "rewrite_queries",  # answer didn't answer question -> better retrieval
            "to_regen": "bump_regen",          # answer ok but ungrounded -> regen with same docs
            "to_end": END,
        },
    )

    return g.compile()




# ------------- Dataset bootstrap -------------
def ensure_dataset(dataset_name: str, path: str) -> str:
    """Create or repopulate the LangSmith dataset from questions.json for inputs,
    but use GroundTruth.json for judging (expected answers)."""
    def _load_rows(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _gt_by_id(gt_rows: list[dict]) -> dict[str, str]:
        by_id = {}
        for it in gt_rows:
            qid = str(it.get("id", "")).strip()
            exp = (it.get("expected_answer") or it.get("answer") or "").strip()
            if qid:
                by_id[qid] = exp
        return by_id

    client = Client()
    existing = next(iter(client.list_datasets(dataset_name=dataset_name)), None)
    if existing:
        count = sum(1 for _ in client.list_examples(dataset_id=existing.id))
        if count > 0:
            print(f"[LangSmith] Using existing dataset '{existing.name}' with {count} examples.")
            return existing.name
        else:
            print(f"[LangSmith] Dataset '{existing.name}' exists but has 0 examples. Repopulating it…")
            q_rows = _load_rows(path)
            gt_rows = _load_rows(GROUNDTRUTH_PATH)
            gt_map = _gt_by_id(gt_rows)

            if not isinstance(q_rows, list):
                raise ValueError("questions.json must be a list of objects with fields: id, text")

            inputs, outputs, metadata = [], [], []
            skipped = 0
            for it in q_rows:
                qid = str(it.get("id", "")).strip()
                qtxt = (it.get("text") or "").strip()
                exp = gt_map.get(qid, "")
                if not qid or not qtxt:
                    skipped += 1
                    continue
                inputs.append({"question": qtxt})
                outputs.append({"expected_answer": exp, "id": qid})
                metadata.append({"eval_id": qid})

            if not inputs:
                raise ValueError("No valid examples in questions.json")

            client.create_examples(
                dataset_id=existing.id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            print(f"[LangSmith] Repopulated '{existing.name}' with {len(inputs)} examples (skipped {skipped}).")
            return existing.name

    q_rows = _load_rows(path)
    gt_rows = _load_rows(GROUNDTRUTH_PATH)
    gt_map = _gt_by_id(gt_rows)

    if not isinstance(q_rows, list):
        raise ValueError("questions.json must be a list of objects with fields: id, text")

    ds = client.create_dataset(dataset_name=dataset_name, description="Eval: Basic RAG over single Weaviate collection (LangGraph)")
    inputs, outputs, metadata = [], [], []
    skipped = 0
    for it in q_rows:
        qid = str(it.get("id", "")).strip()
        qtxt = (it.get("text") or "").strip()
        exp = gt_map.get(qid, "")
        if not qid or not qtxt:
            skipped += 1
            continue
        inputs.append({"question": qtxt})
        outputs.append({"expected_answer": exp, "id": qid})
        metadata.append({"eval_id": qid})

    if not inputs:
        raise ValueError("No valid examples in questions.json")

    client.create_examples(
        dataset_id=ds.id,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )
    print(f"[LangSmith] Created dataset '{ds.name}' with {len(inputs)} examples (skipped {skipped}).")
    return ds.name


# ------------- Bootstrap (build once) -------------
def bootstrap_stack():
    global _client_wv, _global_retriever, _APP
    if _APP is not None:
        return
    kb = read_kb(KB_PATH)
    _client_wv = connect_weaviate_local()
    _global_retriever = build_global_retriever(kb, _client_wv)
    _APP = make_app()

@atexit.register
def _close_weaviate_on_exit():
    try:
        if _client_wv:
            _client_wv.close()
    except Exception:
        pass


# ------------- Predict target for evaluation (via LangGraph) -------------
@traceable(name="predict_basic_rag_graph")
def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    bootstrap_stack()
    q = inputs["question"]
    state = _APP.invoke({
        "question": q,
        "queries": [],
        "parents": [],
        "answer": "",
        "retrieval_ok": False,
        "grounded": True,
        "answer_ok": True,
        "retries_retrieval": 0,
        "retries_generation": 0,
    })
    parents = state.get("parents", [])
    answer = state.get("answer", "")
    return {
        "answer": answer,
        "context": format_ctx(parents),
        "documents": parents,
        "used_categories": ["GLOBAL"],  # parity with earlier evals
        "fused_parent_ids": [d.metadata.get("id") for d in parents],
    }


# ------------- RAG-style graders (same as before) -------------
class CorrectnessGrade(BaseModel):
    explanation: str = Field(..., description="Explain the grade.")
    correct: bool = Field(..., description="True if prediction matches the ground truth.")

class RelevanceGrade(BaseModel):
    explanation: str = Field(..., description="Explain the grade.")
    relevant: bool = Field(..., description="True if answer addresses the question.")

class GroundedGrade(BaseModel):
    explanation: str = Field(..., description="Explain the grade.")
    grounded: bool = Field(..., description="True if answer is supported by the facts.")

class RetrievalRelevanceGrade(BaseModel):
    explanation: str = Field(..., description="Explain the grade.")
    relevant: bool = Field(..., description="True if facts relate to the question.")

def _to_dict(x):
    if hasattr(x, "model_dump"):      # Pydantic v2 models
        return x.model_dump()
    if hasattr(x, "dict"):            # fallback if a v1 model slips in
        return x.dict()
    return x if isinstance(x, dict) else {}

# ---- Correctness (prediction vs dataset reference) ----
correctness_instructions = (
    "You are a teacher grading a quiz. You will be given a QUESTION, the GROUND "
    "TRUTH (correct) ANSWER, and the STUDENT ANSWER. "
    "Grade ONLY factual accuracy relative to the ground truth and ensure there are no "
    "conflicting statements. Extra correct info is fine. "
    "Return your reasoning and a True/False judgment."
)
correctness_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(CorrectnessGrade)

def correctness_eval(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> Dict:
    ref = reference_outputs.get("expected_answer") or reference_outputs.get("answer") or ""
    prompt = (
        f"QUESTION: {inputs.get('question','')}\n"
        f"GROUND TRUTH ANSWER: {ref}\n"
        f"STUDENT ANSWER: {outputs.get('answer','')}"
    )
    grade = _to_dict(correctness_llm.invoke(
        [{"role": "system", "content": correctness_instructions},
         {"role": "user", "content": prompt}]
    ))
    if not str(ref).strip():
        return {"key": "correctness", "score": None, "explanation": "No expected_answer; skipped."}
    score = 1.0 if bool(grade.get("correct")) else 0.0
    return {"key": "correctness", "score": score, "explanation": grade.get("explanation", "")}

# ---- Relevance (answer vs question) ----
relevance_instructions = (
    "You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. "
    "Judge whether the answer is concise and directly addresses the question. "
    "Return your reasoning and a True/False judgment."
)
relevance_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(RelevanceGrade)

def relevance_eval(inputs: Dict, outputs: Dict) -> Dict:
    prompt = f"QUESTION: {inputs.get('question','')}\nSTUDENT ANSWER: {outputs.get('answer','')}"
    grade = _to_dict(relevance_llm.invoke(
        [{"role": "system", "content": relevance_instructions},
         {"role": "user", "content": prompt}]
    ))
    score = 1.0 if bool(grade.get("relevant")) else 0.0
    return {"key": "relevance", "score": score, "explanation": grade.get("explanation", "")}

# ---- Groundedness (answer vs retrieved docs) ----
grounded_instructions = (
    "You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. "
    "Judge whether the answer is fully supported by the FACTS and contains no unsupported claims. "
    "Return your reasoning and a True/False judgment."
)
grounded_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(GroundedGrade)

def groundedness_eval(inputs: Dict, outputs: Dict) -> Dict:
    docs = outputs.get("documents") or []
    facts = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    prompt = f"FACTS:\n{facts}\n\nSTUDENT ANSWER:\n{outputs.get('answer','')}"
    grade = _to_dict(grounded_llm.invoke(
        [{"role": "system", "content": grounded_instructions},
         {"role": "user", "content": prompt}]
    ))
    if not facts.strip():
        return {"key": "groundedness", "score": None, "explanation": "No retrieved context; skipped."}
    score = 1.0 if bool(grade.get("grounded")) else 0.0
    return {"key": "groundedness", "score": score, "explanation": grade.get("explanation", "")}

# ---- Retrieval relevance (facts vs question) ----
retrieval_instructions = (
    "You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS. "
    "Mark True if the facts contain any keywords or semantic meaning related to the question, "
    "False if completely unrelated. Return your reasoning plus the judgment."
)
retrieval_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(RetrievalRelevanceGrade)

def retrieval_relevance_eval(inputs: Dict, outputs: Dict) -> Dict:
    docs = outputs.get("documents") or []
    facts = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    prompt = f"QUESTION: {inputs.get('question','')}\n\nFACTS:\n{facts}"
    grade = _to_dict(retrieval_llm.invoke(
        [{"role": "system", "content": retrieval_instructions},
         {"role": "user", "content": prompt}]
    ))
    if not facts.strip():
        return {"key": "retrieval_relevance", "score": None, "explanation": "No retrieved context; skipped."}
    score = 1.0 if bool(grade.get("relevant")) else 0.0
    return {"key": "retrieval_relevance", "score": score, "explanation": grade.get("explanation", "")}


def _iso_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def dump_answers_to_json(questions_path: str, out_path: str):
    """Run predict(question) for every question and write results to a single JSON file."""
    bootstrap_stack()  # build retriever/app once

    with open(questions_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("questions.json must be a list")

    out = []
    for it in rows:
        qid = str(it.get("id", "")).strip()
        qtxt = (it.get("text") or "").strip()
        if not qid or not qtxt:
            continue
        res = predict({"question": qtxt})  # uses the global _APP/_retriever
        out.append({
            "id": qid,
            "text": qtxt,
            "answer": res.get("answer", ""),
            "context": res.get("context", ""),
            "used_categories": res.get("used_categories", []),
            "fused_parent_ids": res.get("fused_parent_ids", []),
            "generated_at": _iso_now(),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[answers] Wrote {len(out)} answers -> {out_path}")





# ------------- Main -------------
if __name__ == "__main__":
    # Ensure dataset, show count
    if os.getenv("DUMP_ANSWERS", "0") == "1":
        dump_answers_to_json(QUESTIONS_PATH, "answers.json")
        #raise SystemExit(0)
    DATASET_NAME = ensure_dataset(DATASET_NAME, QUESTIONS_PATH)
    client = Client()
    ds = next(iter(client.list_datasets(dataset_name=DATASET_NAME)))
    n = sum(1 for _ in client.list_examples(dataset_id=ds.id))
    print(f"[LangSmith] Ready to evaluate on {n} examples from '{DATASET_NAME}'.")

    # Build once (idempotent)
    bootstrap_stack()

    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[
            correctness_eval,
            relevance_eval,
            groundedness_eval,
            retrieval_relevance_eval,
        ],
        experiment_prefix="answer-quality-rag-graph",
        metadata={
            "gen_model": GEN_MODEL,
            "embed_model": EMBED_MODEL,
            "index": GLOBAL_INDEX,
            "top_k": TOP_K,
            "dataset_file": QUESTIONS_PATH,
        },
        max_concurrency=6,
        upload_results=True,
    )
    print("Experiment:", results.experiment_name)
