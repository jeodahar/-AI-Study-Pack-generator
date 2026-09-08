# workflow.py

import os
import time
from typing import Callable, Dict, Optional

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_RETRIES = 2


class WorkflowError(Exception):
    """Custom error for AI workflow failures."""
MAX_RETRIES = 3

# ============================================================
# GEMINI CLIENT
# ============================================================

def get_api_key() -> str:
    """
    Get Gemini API key from:
    1. Environment variable - useful for Colab/local development
    2. Streamlit Secrets - useful for Streamlit deployment
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]

    except Exception:
        pass

    raise WorkflowError(
        "GEMINI_API_KEY is missing. "
        "Add it to Colab environment variables or Streamlit Secrets."
    )


def get_client():
    """Create and return Gemini client."""

    try:
        return genai.Client(api_key=get_api_key())

    except Exception as exc:
        raise WorkflowError(
            f"Unable to initialize Gemini client: {exc}"
        )


# ============================================================
# GENERIC AI CALL WITH RETRY
# ============================================================

def call_gemini(
    stage: str,
    prompt: str,
    temperature: float = 0.4,
    retries: int = MAX_RETRIES,
) -> str:
    """
    Send a prompt to Gemini with retry and error handling.
    """

    client = get_client()

    last_error = None

    for attempt in range(retries + 1):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": temperature,
                },
            )

            if not response or not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as exc:

            last_error = exc

            if attempt < retries:
                time.sleep(2 * (attempt + 1))

    raise WorkflowError(
        f"{stage} failed after {retries + 1} attempts: "
        f"{last_error}"
    )


# ============================================================
# COMMON SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are an expert educational AI assistant.

Your job is to create accurate, structured and useful
educational material.

Important rules:

1. Follow the requested education level.
2. Follow the requested difficulty.
3. Follow the requested study purpose.
4. Use clear and simple explanations.
5. Use Markdown formatting.
6. Do not invent references or citations.
7. Do not introduce irrelevant topics.
8. Avoid duplicate information.
9. Keep the material exam-oriented when requested.
10. If numerical examples are appropriate, show the solution clearly.
"""


# ============================================================
# STAGE 1 — PLANNER
# ============================================================

def planner(user: Dict) -> str:

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

You are Stage 1: THE STUDY PLANNER.

Your task is NOT to write the complete study notes.

Create a personalized learning plan based on:

Subject:
{user["subject"]}

Topic:
{user["topic"]}

Education Level:
{user["level"]}

Study Purpose:
{user["exam_type"]}

Difficulty:
{user["difficulty"]}

Language:
{user["language"]}

Number of Practice Questions:
{user["question_count"]}

Create the following:

# Study Plan

## 1. Topic Scope

Explain exactly what should be covered.

## 2. Learning Objectives

Create 5–8 measurable learning objectives.

## 3. Core Subtopics

List the most important subtopics in logical order.

## 4. Recommended Study Sequence

Explain the order in which the learner should study.

## 5. Important Skills

Identify what the learner should be able to explain,
calculate, compare, analyze or solve.

## 6. Exam Priorities

Identify the most important concepts for the selected
study purpose.

## 7. Assessment Strategy

Suggest how the requested practice questions should
be distributed.

Do not generate the full study material.
This stage is ONLY planning.
"""

    return call_gemini(
        "Planner",
        prompt,
        temperature=0.3,
    )


# ============================================================
# STAGE 2 — CONTENT GENERATOR
# ============================================================

def content_generator(
    user: Dict,
    plan: str,
) -> str:

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

You are Stage 2: THE CONTENT GENERATOR.

Create comprehensive study material using the
learner requirements and the approved study plan.

========================
LEARNER REQUIREMENTS
========================

Subject:
{user["subject"]}

Topic:
{user["topic"]}

Education Level:
{user["level"]}

Study Purpose:
{user["exam_type"]}

Difficulty:
{user["difficulty"]}

Language:
{user["language"]}

========================
STUDY PLAN
========================

{plan}

========================
CONTENT REQUIREMENTS
========================

Create:

# Study Notes

## 1. Topic Overview

## 2. Learning Objectives

## 3. Key Concepts

## 4. Important Definitions

## 5. Detailed Explanation

## 6. Examples

Include worked examples where appropriate.

## 7. Important Facts / Formulas

Include formulas only when relevant.

## 8. Common Mistakes

## 9. Exam-Focused Points

## 10. Quick Revision Notes

Important:

Follow the study plan closely.

Do not generate practice questions in this stage.

Do not discuss the internal AI workflow.
"""

    return call_gemini(
        "Content Generator",
        prompt,
        temperature=0.45,
    )


# ============================================================
# STAGE 3 — ASSESSMENT GENERATOR
# ============================================================

def assessment_generator(
    user: Dict,
    plan: str,
    content: str,
) -> str:

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

You are Stage 3: THE ASSESSMENT GENERATOR.

Create an assessment based ONLY on the material
covered in the study content.

========================
LEARNER
========================

Subject:
{user["subject"]}

Topic:
{user["topic"]}

Education Level:
{user["level"]}

Study Purpose:
{user["exam_type"]}

Difficulty:
{user["difficulty"]}

Language:
{user["language"]}

Number of Questions:
{user["question_count"]}

========================
STUDY PLAN
========================

{plan}

========================
STUDY CONTENT
========================

{content}

========================
ASSESSMENT
========================

Create exactly {user["question_count"]} questions.

Use a suitable mixture of:

- Multiple Choice Questions
- Short Answer Questions
- Conceptual Questions
- Numerical / Problem-Solving Questions when appropriate

For every question provide:

### Question

### Answer

### Explanation

Rules:

1. Questions must be based on the actual content.
2. Avoid duplicate questions.
3. Avoid ambiguous questions.
4. Make difficulty appropriate.
5. Make sure every answer is consistent with the content.
6. Use numerical questions only when relevant to the subject.
"""

    return call_gemini(
        "Assessment Generator",
        prompt,
        temperature=0.35,
    )


# ============================================================
# STAGE 4 — REVIEWER
# ============================================================

def reviewer(
    user: Dict,
    plan: str,
    content: str,
    assessment: str,
) -> str:

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

You are Stage 4: THE QUALITY REVIEWER.

Your job is to critically evaluate the generated
study pack before it reaches the learner.

========================
LEARNER REQUIREMENTS
========================

{user}

========================
STUDY PLAN
========================

{plan}

========================
CONTENT
========================

{content}

========================
ASSESSMENT
========================

{assessment}

========================
REVIEW CHECKLIST
========================

Evaluate:

1. Factual accuracy
2. Completeness
3. Learning objective coverage
4. Education-level suitability
5. Difficulty suitability
6. Exam relevance
7. Explanation clarity
8. Formula correctness
9. Example correctness
10. Question quality
11. Answer-key correctness
12. Internal consistency
13. Missing important concepts
14. Unnecessary or irrelevant content

Return:

# Review Report

## Overall Status

Choose:

PASS

or

NEEDS REVISION

## Strengths

List what is already good.

## Problems Found

List specific problems.

## Corrections Required

Explain exactly what should be corrected.

## Refinement Priorities

Rank the most important improvements.

Do NOT rewrite the complete study pack.
Only provide the review and correction instructions.
"""

    return call_gemini(
        "Reviewer",
        prompt,
        temperature=0.2,
    )


# ============================================================
# STAGE 5 — REFINER
# ============================================================

def refiner(
    user: Dict,
    plan: str,
    content: str,
    assessment: str,
    review: str,
) -> str:

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

You are Stage 5: THE FINAL REFINER.

Create the final personalized study pack.

You have received:

1. Original learner requirements
2. Study plan
3. Generated content
4. Assessment
5. Quality review

Your responsibility is to improve the material using
the review feedback.

========================
LEARNER REQUIREMENTS
========================

{user}

========================
STUDY PLAN
========================

{plan}

========================
ORIGINAL CONTENT
========================

{content}

========================
ASSESSMENT
========================

{assessment}

========================
REVIEW REPORT
========================

{review}

========================
FINAL STUDY PACK
========================

Create one polished Markdown document.

Use this structure:

# AI Study Pack

## 1. Topic Overview

## 2. Learning Objectives

## 3. Key Concepts

## 4. Important Definitions

## 5. Detailed Study Notes

## 6. Examples / Worked Problems

## 7. Important Facts and Formulas

## 8. Common Mistakes

## 9. Exam-Focused Points

## 10. Quick Revision Notes

## 11. Practice Questions

## 12. Answer Key with Explanations

## 13. Final Self-Test

Create 5 short self-test questions.

## 14. Revision Checklist

Create a concise checklist.

========================
REFINEMENT RULES
========================

1. Correct all valid issues identified by the reviewer.
2. Preserve accurate information.
3. Improve weak explanations.
4. Remove unnecessary repetition.
5. Ensure assessment answers are correct.
6. Ensure the final material matches the requested level.
7. Ensure the difficulty is appropriate.
8. Ensure the selected language is respected.
9. Keep the study pack organized.
10. Do not mention the AI workflow.
11. Do not mention internal prompts.
12. Do not mention the reviewer.
13. Do not invent citations.
"""

    return call_gemini(
        "Refiner",
        prompt,
        temperature=0.35,
    )


# ============================================================
# COMPLETE WORKFLOW
# ============================================================

def run_study_workflow(
    user_input: Dict,
    progress_callback: Optional[
        Callable[[str, int], object]
    ] = None,
) -> Dict[str, str]:
    """
    Execute the complete AI workflow:

    Planner
       ↓
    Content Generator
       ↓
    Assessment Generator
       ↓
    Reviewer
       ↓
    Refiner
    """

    def update(message: str, percent: int):

        if progress_callback:
            progress_callback(message, percent)

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    update(
        "🔄 Stage 1/5 — Planning personalized learning objectives...",
        10,
    )

    plan = planner(user_input)

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    update(
        "🔄 Stage 2/5 — Generating study content...",
        30,
    )

    content = content_generator(
        user_input,
        plan,
    )

    # --------------------------------------------------------
    # STAGE 3
    # --------------------------------------------------------

    update(
        "🔄 Stage 3/5 — Creating assessment questions...",
        50,
    )

    assessment = assessment_generator(
        user_input,
        plan,
        content,
    )

    # --------------------------------------------------------
    # STAGE 4
    # --------------------------------------------------------

    update(
        "🔄 Stage 4/5 — Reviewing accuracy and quality...",
        70,
    )

    review = reviewer(
        user_input,
        plan,
        content,
        assessment,
    )

    # --------------------------------------------------------
    # STAGE 5
    # --------------------------------------------------------

    update(
        "🔄 Stage 5/5 — Refining the final study pack...",
        85,
    )

    final_pack = refiner(
        user_input,
        plan,
        content,
        assessment,
        review,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    update(
        "✅ All five AI workflow stages completed successfully.",
        100,
    )

    return {
        "plan": plan,
        "content": content,
        "assessment": assessment,
        "review": review,
        "final_pack": final_pack,
    }
