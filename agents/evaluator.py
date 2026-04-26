"""
Evaluator Agent
────────────────
Uses the Hybrid Scoring Engine:
  - LLM score      (50%) — qualitative depth assessment
  - Semantic score (30%) — ChromaDB distance vs gold-standard answer
  - Keyword score  (20%) — technical term coverage
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from tools.scoring import compute_hybrid_score, score_to_proficiency
from .state import CatalystState, SkillEvaluation

# Lazy-loaded LLM — created on first call, not at import time
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return _llm

_SYSTEM_PROMPT = """
You are an expert technical assessor. You evaluate interview answers strictly
and objectively. You score from 0 to 5:
  0 = No knowledge / completely wrong
  1 = Aware of concept but no real experience
  2 = Basic usage, limited depth
  3 = Solid understanding, some gaps
  4 = Strong practical experience
  5 = Expert-level, deep nuanced knowledge
Provide a score and concise reasoning (2 sentences max).
"""


class _LLMEvaluation(BaseModel):
    score: float = Field(description="Score from 0.0 to 5.0")
    reasoning: str = Field(description="Brief explanation for the score")


def run(state: CatalystState) -> dict:
    skills = state.get("skills_to_assess", [])
    idx = state.get("current_skill_index", 0)

    if idx >= len(skills):
        return {}

    current_skill = skills[idx]
    messages = state.get("messages", [])

    # Extract the last Q&A pair
    context = messages[-2:] if len(messages) >= 2 else messages
    if not context:
        return {}

    # Reconstruct answer text from last HumanMessage
    answer_text = ""
    for m in reversed(context):
        if isinstance(m, HumanMessage):
            answer_text = m.content
            break

    # 1. Get LLM qualitative score
    structured_llm = _get_llm().with_structured_output(_LLMEvaluation)
    eval_prompt = (
        f"Skill being assessed: {current_skill}\n\n"
        f"Interview conversation (last 2 turns):\n"
        + "\n".join(f"{type(m).__name__}: {m.content}" for m in context)
    )
    llm_result = structured_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=eval_prompt)
    ])

    # 2. Hybrid scoring (adds semantic + keyword signals)
    scores = compute_hybrid_score(llm_result.score, current_skill, answer_text)
    proficiency = score_to_proficiency(scores["final_score"])

    evaluation = SkillEvaluation(
        skill=current_skill,
        llm_score=scores["llm_score"],
        semantic_score=scores["semantic_score"],
        keyword_score=scores["keyword_score"],
        final_score=scores["final_score"],
        proficiency=proficiency,
        reasoning=llm_result.reasoning,
    )

    existing = list(state.get("evaluations", []))
    existing.append(evaluation)

    return {
        "evaluations": existing,
        "current_skill_index": idx + 1,
    }
