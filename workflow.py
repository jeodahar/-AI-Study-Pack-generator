import os
import time
from typing import Callable, Dict, Optional
from google import genai
from groq import Groq


class WorkflowError(Exception):
    """Raised when an AI workflow stage cannot be completed."""


# Keep the model in one place so it can be changed easily.
GEMINI_MODEL = "gemini-2.5-flash"
MODEL = os.getenv("GROQ_MODEL", "gpt-oss-120b")


def _get_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise WorkflowError(
            "GROQ_API_KEY is missing. Add it to your Colab environment or Streamlit Secrets."
        )
    return api_key


def _client() -> Groq:
    return Groq(api_key=_get_api_key())


def _call_ai(
    stage: str,
    prompt: str,
    retries: int = 2,
    temperature: float = 0.4,
) -> str:
    """Call Groq with bounded retries and clear stage-specific errors."""
    client = _client()
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful educational AI. Follow the requested "
                            "education level, purpose, difficulty, and language. "
                            "Do not invent sources or citations. Produce useful, "
                            "well-structured Markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ValueError("The model returned an empty response.")
            return text.strip()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise WorkflowError(
        f"{stage} failed after {retries + 1} attempts: {last_error}"
    )


def planner(user: Dict) -> str:
    return _call_ai(
        "Planner",
        f"""
Create a personalized study plan for this learner.

Subject: {user['subject']}
Topic: {user['topic']}
Education level: {user['level']}
Study purpose: {user['exam_type']}
Difficulty: {user['difficulty']}
Language: {user['language']}
Practice questions requested: {user['question_count']}

Return Markdown with:
1. Topic scope
2. 5-8 measurable learning objectives
3. Recommended study sequence
4. Core subtopics
5. Skills the learner should demonstrate
6. Exam/revision priorities
7. Suggested question distribution

Do not write the full notes yet. This stage is planning only.
""",
        temperature=0.3,
    )


def content_generator(user: Dict, plan: str) -> str:
    return _call_ai(
        "Content Generator",
        f"""
Generate the study content using the learner requirements and the approved plan.

LEARNER REQUIREMENTS:
{user}

STUDY PLAN:
{plan}

Create:
- Topic overview
- Learning objectives
- Key concepts and definitions
- Detailed explanations
- Examples or worked examples where appropriate
- Important formulas/facts where applicable
- Common mistakes
- Quick revision notes
- Exam-focused points

Stay within the planned scope and match the requested level and difficulty.
Use Markdown. Do not create practice questions yet.
""",
        temperature=0.45,
    )


def assessment_generator(user: Dict, plan: str, content: str) -> str:
    return _call_ai(
        "Assessment Generator",
        f"""
Create an assessment from the study material below.

LEARNER REQUIREMENTS:
{user}

STUDY PLAN:
{plan}

STUDY CONTENT:
{content}

Generate exactly {user['question_count']} practice questions.
Use a balanced mixture appropriate to the subject:
- MCQs when suitable
- Short-answer questions
- Numerical/problem-solving questions when suitable

For every question provide the correct answer and a concise explanation.
Questions must test concepts actually covered in the content.
Avoid duplicate questions and ambiguous answers.
""",
        temperature=0.35,
    )


def reviewer(user: Dict, plan: str, content: str, assessment: str) -> str:
    return _call_ai(
        "Reviewer",
        f"""
Act as a strict educational quality reviewer.

LEARNER REQUIREMENTS:
{user}

STUDY PLAN:
{plan}

CONTENT:
{content}

ASSESSMENT:
{assessment}

Check:
1. Factual accuracy
2. Coverage of learning objectives
3. Appropriate difficulty and education level
4. Relevance to the study purpose
5. Internal consistency
6. Assessment-answer correctness
7. Clarity and organization
8. Missing or weak explanations

Return:
- Overall status: PASS or NEEDS REVISION
- Strengths
- Specific issues
- Corrections required
- Prioritized refinement instructions

Do not rewrite the entire study pack.
""",
        temperature=0.2,
    )


def refiner(
    user: Dict,
    plan: str,
    content: str,
    assessment: str,
    review: str,
) -> str:
    return _call_ai(
        "Refiner",
        f"""
Produce the final personalized study pack.

LEARNER REQUIREMENTS:
{user}

ORIGINAL STUDY PLAN:
{plan}

GENERATED CONTENT:
{content}

ASSESSMENT:
{assessment}

REVIEW REPORT:
{review}

Apply every valid correction identified by the reviewer.
Preserve correct material, improve weak areas, and ensure consistency.

Return one polished Markdown document with:
# AI Study Pack
## 1. Topic Overview
## 2. Learning Objectives
## 3. Key Concepts and Definitions
## 4. Detailed Study Notes
## 5. Examples / Worked Problems
## 6. Important Facts or Formulas
## 7. Common Mistakes
## 8. Quick Revision Notes
## 9. Practice Questions
## 10. Answer Key with Explanations
## 11. Final Self-Test
## 12. Revision Checklist

Match the requested language, education level, difficulty, and study purpose.
Do not mention this workflow, internal prompts, or reviewer instructions in the final document.
""",
        temperature=0.35,
    )


def run_study_workflow(
    user_input: Dict,
    progress_callback: Optional[Callable[[str, int], object]] = None,
) -> Dict[str, str]:
    """Run the sequential Plan → Generate → Assess → Review → Refine workflow."""

    def update(message: str, percent: int) -> None:
        if progress_callback:
            progress_callback(message, percent)

    update("🔄 Stage 1/5 — Planning personalized learning objectives...", 10)
    plan = planner(user_input)

    update("🔄 Stage 2/5 — Generating study content from the plan...", 30)
    content = content_generator(user_input, plan)

    update("🔄 Stage 3/5 — Creating assessment from the generated content...", 50)
    assessment = assessment_generator(user_input, plan, content)

    update("🔄 Stage 4/5 — Reviewing accuracy, completeness, and assessment quality...", 70)
    review = reviewer(user_input, plan, content, assessment)

    update("🔄 Stage 5/5 — Refining the study pack using reviewer feedback...", 85)
    final_pack = refiner(user_input, plan, content, assessment, review)

    update("✅ Workflow complete — final study pack generated.", 100)

    return {
        "plan": plan,
        "content": content,
        "assessment": assessment,
        "review": review,
        "final_pack": final_pack,
    }
