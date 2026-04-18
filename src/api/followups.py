from flask import Blueprint, request, jsonify, current_app
from src.services.groq_service import GroqService

followups_bp = Blueprint("followups", __name__)

@followups_bp.post("/followups")
def followups():
    data     = request.get_json(silent=True) or {}
    question = data.get("question", "")
    answer   = data.get("answer", "")

    if not question:
        return jsonify({"questions": []}), 200

    messages = [
        {
            "role": "system",
            "content": "You are a utility that outputs only raw JSON arrays. No explanation. No markdown."
        },
        {
            "role": "user",
            "content": (
                f"Given this NEET study Q&A, return exactly 3 follow-up questions "
                f"a student would ask next.\n"
                f"Output format: [\"q1\", \"q2\", \"q3\"]\n\n"
                f"Question: {question}\n"
                f"Answer: {answer[:300]}"
            )
        }
    ]

    cfg = current_app.config
    llm = GroqService(
        api_key=cfg["GROQ_API_KEY"],
        model=cfg["GROQ_MODEL"],
        max_tokens=150,           # ← tiny, just 3 short strings
        temperature=0.7
    )

    raw = llm.complete(messages=messages)

    try:
        import re, json
        match = re.search(r'\[[\s\S]*?\]', raw)
        questions = json.loads(match.group()) if match else []
    except Exception:
        questions = []

    return jsonify({"questions": questions}), 200