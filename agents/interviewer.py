"""
Interviewer Agent
──────────────────
Generates one open-ended, scenario-based interview question per skill.
Each question targets real-world experience, not trivia.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import CatalystState

# Lazy-loaded LLM — created on first call, not at import time
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)
    return _llm

_SYSTEM_PROMPT = """
You are a Staff Engineer conducting a technical interview.
Your style is conversational, respectful, and focused on real-world scenarios.
You ask ONE concise question at a time (2–3 sentences max).
You NEVER ask trivia or definitions. You probe for hands-on experience.
Ask directly — no preamble, no "Sure!" or "Great!".
"""


def run(state: CatalystState) -> dict:
    skills = state.get("skills_to_assess", [])
    idx = state.get("current_skill_index", 0)

    if idx >= len(skills):
        return {"interview_complete": True}

    current_skill = skills[idx]
    messages = state.get("messages", [])

    # Build context prompt
    skill_prompt = (
        f"Ask a scenario-based interview question to assess the candidate's "
        f"REAL hands-on experience with: **{current_skill}**. "
        f"This is question #{idx + 1} of {len(skills)}."
    )

    # Include recent conversation for context (avoid stale history)
    recent = messages[-4:] if len(messages) >= 4 else messages
    # Ensure there's at least one human-turn message for Gemini
    if not any(isinstance(m, HumanMessage) for m in recent):
        recent = [HumanMessage(content="I am ready for my next question.")]

    response = _get_llm().invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=skill_prompt)] + recent
    )

    return {"messages": [response]}
