"""
Skill Extractor Agent
─────────────────────
First checks the NetworkX Skill Knowledge Graph to deterministically
find skills mentioned in the JD. Then uses LLM only for skills the
graph doesn't know about (fallback).
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from data.skill_graph import find_skills_in_text, SKILLS
from .state import CatalystState

# Lazy-loaded LLM — created on first call, not at import time
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    return _llm

_SYSTEM_PROMPT = """
You are a Senior Technical Recruiter specialising in skill gap analysis.
Your only job is to read a Job Description and a Resume, then identify
the 3-5 MOST IMPORTANT technical skills required by the JD that must be
assessed in a technical interview. Prioritise skills where the resume
shows shallow or no evidence of real experience.
Output ONLY a JSON array of skill names, e.g. ["Django", "PostgreSQL", "Docker"].
"""


class _ExtractedSkills(BaseModel):
    skills: list[str] = Field(description="List of 3-5 core skills to assess")


def run(state: CatalystState) -> dict:
    jd = state.get("job_description", "")
    resume = state.get("resume_text", "")

    # Step 1: Graph-based extraction (deterministic, free)
    jd_skills = find_skills_in_text(jd)
    resume_skills = find_skills_in_text(resume)
    graph_gaps = [s for s in jd_skills if s not in resume_skills]

    # Step 2: If graph finds enough skills, skip LLM call
    if len(graph_gaps) >= 3:
        skills = graph_gaps[:5]
    else:
        # LLM fallback for skills the graph doesn't know
        structured_llm = _get_llm().with_structured_output(_ExtractedSkills)
        prompt = f"{_SYSTEM_PROMPT}\n\nJOB DESCRIPTION:\n{jd}\n\nRESUME:\n{resume}"
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        llm_skills = result.skills

        # Merge graph findings with LLM findings, deduplicate
        combined = list(dict.fromkeys(graph_gaps + llm_skills))
        skills = combined[:5]

    return {"skills_to_assess": skills, "current_skill_index": 0}
