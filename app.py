import os
import re
import markdown as md_lib
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from query import ask_question

load_dotenv()

app = Flask(__name__)

# Maps each section emoji to (background, border-color, extra-css-class)
_SECTION_STYLES = {
    "📋": ("#eff6ff", "#3b82f6", ""),
    "📝": ("#f5f3ff", "#7c3aed", ""),
    "🎯": ("#ecfdf5", "#059669", ""),
    "⚠️": ("#fffbeb", "#d97706", ""),
    "✅": ("#f0fdf4", "#16a34a", "section-model-answer"),
}

# Matches: emoji  **TITLE**\n  body  (until next section or end)
_SECTION_RE = re.compile(
    r'^(📋|📝|🎯|⚠️|✅)\s+\*\*(.+?)\*\*[ \t]*\n(.*?)(?=\n(?:📋|📝|🎯|⚠️|✅)\s+\*\*|\Z)',
    re.MULTILINE | re.DOTALL,
)


def render_answer_html(answer: str) -> str:
    parts = []
    for match in _SECTION_RE.finditer(answer):
        emoji = match.group(1)
        title = match.group(2).strip()
        content = match.group(3).strip()
        bg, border, extra_cls = _SECTION_STYLES.get(emoji, ("#f9fafb", "#6b7280", ""))
        content_html = md_lib.markdown(content)
        cls = f"answer-section {extra_cls}".strip()
        parts.append(
            f'<div class="{cls}" style="background:{bg};border-left:4px solid {border}">'
            f'<div class="section-header">{emoji} {title}</div>'
            f'<div class="section-body">{content_html}</div>'
            f'</div>'
        )
    # Fallback: plain markdown if no sections matched
    return "\n".join(parts) if parts else md_lib.markdown(answer)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        result = ask_question(question)
        result["answer"] = render_answer_html(result["answer"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
