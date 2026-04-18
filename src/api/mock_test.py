"""
src/api/mock_test.py — Mock Test API Endpoints
================================================
Two endpoints:

  POST /api/mock/generate
  ─────────────────────────
  Generates a set of MCQs using Groq LLM with an NCERT-aware prompt.
  Returns structured JSON with questions, options, correct answer, explanation.

  Request:
    {
      "topic":       "Cell Cycle and Division",  ← specific topic OR "Full NEET"
      "subject":     "Biology",                  ← Biology | Physics | Chemistry | Mixed
      "count":       10,                         ← number of questions (5–180)
      "difficulty":  "Medium"                    ← Easy | Medium | Hard | Mixed
    }

  Response:
    {
      "test_id":   "uuid",
      "topic":     "Cell Cycle and Division",
      "questions": [
        {
          "id": 1,
          "question": "...",
          "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
          "correct_index": 2,          ← 0-based index of correct option
          "explanation": "...",
          "subject": "Biology",
          "chapter": "Cell Cycle and Cell Division"
        },
        ...
      ]
    }

  POST /api/mock/explain
  ─────────────────────────
  Generates AI explanation for a single question — used on the results page
  for wrong answers and marked-for-review questions.

  Request:
    {
      "question":       "...",
      "options":        ["A...", "B...", "C...", "D..."],
      "correct_index":  2,
      "student_index":  1,        ← what the student chose (-1 if unanswered)
      "subject":        "Biology"
    }

  Response:
    { "explanation": "..." }
"""

import uuid
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app
from pydantic import BaseModel, ValidationError, field_validator

from src.services.groq_service import GroqService

mock_bp = Blueprint("mock", __name__)
logger = logging.getLogger(__name__)


# ── Request Models ─────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic:      str    = "Full NEET"
    subject:    str    = "Mixed"
    count:      int    = 10
    difficulty: str    = "Mixed"

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v):
        allowed = {"Biology", "Physics", "Chemistry", "Mixed"}
        return v if v in allowed else "Mixed"

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v):
        allowed = {"Easy", "Medium", "Hard", "Mixed"}
        return v if v in allowed else "Mixed"

    @field_validator("count")
    @classmethod
    def validate_count(cls, v):
        return max(1, min(180, v))  # clamp between 1 and 180


class ExplainRequest(BaseModel):
    question:      str
    options:       list
    correct_index: int
    student_index: int   = -1   # -1 = not answered
    subject:       str   = "Biology"


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_generation_prompt(topic: str, subject: str, count: int, difficulty: str) -> list[dict]:
    """
    Builds the message list for MCQ generation.
    The system prompt instructs the LLM to return ONLY valid JSON.
    """

    # Subject context for more targeted generation
    subject_context = {
        "Biology": "Focus on NCERT Biology Class 11 & 12 — covers Cell Biology, Genetics, Human Physiology, Plant Physiology, Ecology, Evolution, Biotechnology.",
        "Physics": "Focus on NCERT Physics Class 11 & 12 — covers Mechanics, Thermodynamics, Waves, Optics, Electrostatics, Current Electricity, Modern Physics.",
        "Chemistry": "Focus on NCERT Chemistry Class 11 & 12 — covers Physical Chemistry, Organic Chemistry, Inorganic Chemistry, Coordination Compounds, Biomolecules.",
        "Mixed": "Draw from all three NEET subjects: Biology (50% weightage), Physics (25%), Chemistry (25%). Match the real NEET distribution.",
    }.get(subject, "Draw from all three NEET subjects.")

    difficulty_guide = {
        "Easy":   "Questions should be direct recall from NCERT — definitions, one-step reasoning, straightforward identification. A student who has read NCERT once should get 80%+ correct.",
        "Medium": "Questions require understanding and application — multi-step reasoning, comparing two concepts, applying a principle to a scenario. Typical of NEET paper difficulty.",
        "Hard":   "Questions require deep understanding — multi-concept integration, exception-based, data interpretation, or tricky distractors that exploit common misconceptions. High NEET rank territory.",
        "Mixed":  "Mix of all difficulties: approximately 30% Easy, 50% Medium, 20% Hard. This mirrors the actual NEET paper distribution.",
    }.get(difficulty, "Mix of all difficulties.")

    topic_instruction = (
        f"Generate questions specifically about: **{topic}**"
        if topic.lower() not in ("full neet", "neet", "all topics", "")
        else "Generate questions across the full NEET syllabus. Distribute topics naturally — don't cluster."
    )

    system_prompt = f"""You are a NEET question paper setter with 15 years of experience.
Your job: generate {count} high-quality NEET-style MCQs.

STRICT OUTPUT RULE:
Return ONLY a valid JSON array. No preamble, no explanation, no markdown fences.
Start your response with [ and end with ].

QUESTION FORMAT (each item in the array):
{{
  "id": <integer, 1-based>,
  "question": "<question text — clear, unambiguous, NEET-style>",
  "options": ["<option A text>", "<option B text>", "<option C text>", "<option D text>"],
  "correct_index": <0-3, which option is correct>,
  "explanation": "<2-3 sentence explanation of why the correct answer is right and why the distractors are wrong. NCERT-grounded.>",
  "subject": "<Biology|Physics|Chemistry>",
  "chapter": "<NCERT chapter name>"
}}

QUALITY RULES:
- Each question must have EXACTLY 4 options (A, B, C, D).
- Options must be plausible — not obviously wrong. Good distractors test understanding.
- correct_index must be 0, 1, 2, or 3 (0 = first option, 3 = fourth option).
- No "All of the above" or "None of the above" options — NEET stopped using these.
- Questions must be NCERT-accurate. No invented facts.
- Vary question types: identification, reason-assertion logic, definition, application, comparison.
- Do not repeat the same concept twice in the same test.

{topic_instruction}
Subject scope: {subject_context}
Difficulty: {difficulty_guide}

Generate exactly {count} questions now."""

    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {count} NEET MCQs now. Topic: {topic}. Return only the JSON array."}]


def _build_explanation_prompt(question: str, options: list, correct_index: int,
                               student_index: int, subject: str) -> list[dict]:
    """Builds prompt for explaining a single question to a student."""

    correct_option = options[correct_index] if 0 <= correct_index < len(options) else "N/A"
    student_option = options[student_index] if 0 <= student_index < len(options) else "Not answered"

    outcome = "unanswered" if student_index == -1 else (
        "correct" if student_index == correct_index else "incorrect"
    )

    outcome_line = {
        "correct":    "The student got this RIGHT — but reinforce the concept so it sticks.",
        "incorrect":  f"The student chose: {student_option}. They got this WRONG. Address their specific misconception.",
        "unanswered": "The student left this unanswered. Explain the concept from scratch.",
    }[outcome]

    system = f"""You are AI Neet Mentor — a warm, sharp NEET tutor who explains concepts brilliantly.
A student just finished a mock test. You are now reviewing one question with them.

{outcome_line}

YOUR EXPLANATION STYLE:
- Start with a one-line hook: the simplest version of the answer.
- Explain WHY the correct answer is right — use NCERT reasoning, not just facts.
- If they got it wrong, gently explain why their choice ({student_option}) was incorrect — what trap did they fall into?
- End with a NEET tip: what pattern does NEET use when testing this concept?
- Keep it under 200 words. Dense, not rambling.
- No labels like "Hook:" or "NEET tip:" — just write naturally.
- Subject: {subject}"""

    options_text = "\n".join(f"  {'✓' if i == correct_index else '✗' if i == student_index else '○'} {opt}"
                             for i, opt in enumerate(options))

    user_msg = f"""Question: {question}

Options:
{options_text}

Correct answer: {correct_option}
Student's answer: {student_option}

Explain this to the student now."""

    return [{"role": "system", "content": system},
            {"role": "user", "content": user_msg}]


# ── JSON parser with fallback ──────────────────────────────────────────────────

def _parse_questions_json(raw: str) -> list:
    """
    Robustly extracts JSON from LLM output.
    LLMs sometimes wrap JSON in markdown fences or add preamble.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the outermost array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in LLM response")

    json_str = cleaned[start:end + 1]
    return json.loads(json_str)


def _validate_question(q: dict, idx: int) -> dict:
    """Validates and normalises a single question dict."""
    return {
        "id":            q.get("id", idx + 1),
        "question":      str(q.get("question", "")).strip(),
        "options":       [str(o).strip() for o in q.get("options", [])[:4]],
        "correct_index": max(0, min(3, int(q.get("correct_index", 0)))),
        "explanation":   str(q.get("explanation", "")).strip(),
        "subject":       str(q.get("subject", "Biology")).strip(),
        "chapter":       str(q.get("chapter", "NCERT")).strip(),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@mock_bp.post("/mock/generate")
def generate_mock_test():
    """
    Generates a mock test from config params.
    Uses Groq LLM with an NCERT-aware MCQ generation prompt.
    For large counts (>20), batches into multiple LLM calls and merges.
    """
    raw = request.get_json(silent=True)
    if not raw:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        gen_req = GenerateRequest(**raw)
    except ValidationError as e:
        return jsonify({"error": "Invalid request", "details": [err["msg"] for err in e.errors()]}), 422

    llm = GroqService(
        api_key=current_app.config["GROQ_API_KEY"],
        model=current_app.config["GROQ_MODEL"],
        max_tokens=4096,
        temperature=0.7,
    )

    # ── Batch strategy ───────────────────────────────────────────────────
    # Groq llama-3.1-8b can reliably output ~20 MCQs per call in JSON.
    # For larger tests, split into batches of 20 and merge.
    BATCH_SIZE = 20
    total_needed = gen_req.count
    all_questions = []
    batch_num = 0

    while len(all_questions) < total_needed:
        remaining = total_needed - len(all_questions)
        batch_count = min(BATCH_SIZE, remaining)
        batch_num += 1

        logger.info("Batch %d: generating %d MCQs | topic=%s subject=%s difficulty=%s",
                    batch_num, batch_count, gen_req.topic, gen_req.subject, gen_req.difficulty)

        messages = _build_generation_prompt(
            topic=gen_req.topic,
            subject=gen_req.subject,
            count=batch_count,
            difficulty=gen_req.difficulty,
        )

        raw_response = llm.complete(messages=messages)

        try:
            batch_raw = _parse_questions_json(raw_response)
            batch = [_validate_question(q, len(all_questions) + i) for i, q in enumerate(batch_raw)]
            all_questions.extend(batch)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Batch %d parse failed: %s", batch_num, e)
            # If we have at least some questions, proceed with what we have
            if all_questions:
                break
            return jsonify({
                "error": "Question generation failed — the AI returned malformed data. Please try again.",
                "debug": str(e)
            }), 500

        # Safety: cap at requested count (LLM sometimes generates extra)
        if len(all_questions) >= total_needed:
            all_questions = all_questions[:total_needed]
            break

    if not all_questions:
        return jsonify({"error": "No questions were generated. Please try again."}), 500

    # Re-sequence IDs after merging batches
    for i, q in enumerate(all_questions):
        q["id"] = i + 1

    test_id = str(uuid.uuid4())

    return jsonify({
        "test_id":    test_id,
        "topic":      gen_req.topic,
        "subject":    gen_req.subject,
        "difficulty": gen_req.difficulty,
        "count":      len(all_questions),
        "questions":  all_questions,
    }), 200


@mock_bp.post("/mock/explain")
def explain_question():
    """
    Returns an AI explanation for a single question.
    Called from the results page for wrong + marked-for-review questions.
    """
    raw = request.get_json(silent=True)
    if not raw:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        exp_req = ExplainRequest(**raw)
    except ValidationError as e:
        return jsonify({"error": "Invalid request", "details": [err["msg"] for err in e.errors()]}), 422

    llm = GroqService(
        api_key=current_app.config["GROQ_API_KEY"],
        model=current_app.config["GROQ_MODEL"],
        max_tokens=512,
        temperature=0.4,
    )

    messages = _build_explanation_prompt(
        question=exp_req.question,
        options=exp_req.options,
        correct_index=exp_req.correct_index,
        student_index=exp_req.student_index,
        subject=exp_req.subject,
    )

    explanation = llm.complete(messages=messages)

    return jsonify({"explanation": explanation}), 200
