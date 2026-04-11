import os
import openai
from dotenv import load_dotenv
from utils import embed_text, get_pinecone_index

load_dotenv()

TOP_K = 5  # Number of pages to retrieve per query


def retrieve_context(question: str) -> list[dict]:
    """Embed the question and fetch the most relevant pages from Pinecone."""
    index = get_pinecone_index()
    embedding = embed_text(question)
    results = index.query(vector=embedding, top_k=TOP_K, include_metadata=True)
    return results["matches"]


def build_context_block(matches: list[dict]) -> str:
    parts = []
    for match in matches:
        meta = match["metadata"]
        parts.append(f"[Page {meta['page_number']}]\n{meta['text']}")
    return "\n\n---\n\n".join(parts)


def ask_question(question: str) -> dict:
    """Return a dict with 'answer' and 'pages' keys. Used by both CLI and web."""
    matches = retrieve_context(question)
    if not matches:
        return {"answer": "No relevant content found in the textbook.", "pages": []}

    context = build_context_block(matches)
    page_nums = sorted(m["metadata"]["page_number"] for m in matches)

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""You are a senior Cambridge IGCSE Economics (0455) examiner and tutor. Your job is to teach students how to answer exam questions the way examiners want — using the command word, the mark scheme, and the textbook excerpts below.

TEXTBOOK EXCERPTS:
{context}

STUDENT QUESTION:
{question}

==========================================
STEP 1 — AUTO-DETECT THE QUESTION TYPE
==========================================
Identify the command word in the student's question and classify it:

- "Define / What is / State" → DEFINE (2 marks)
- "Explain / Describe / Give reasons" → EXPLAIN (4 marks)
- "Analyse / Examine / Calculate with explanation" → ANALYSE (6 marks)
- "Discuss / Evaluate / To what extent / Do you agree" → DISCUSS/EVALUATE (8 marks)

If the student already states the mark value (e.g. "6 marks"), use that. Otherwise infer from the command word above.

==========================================
STEP 2 — MARK ALLOCATION RULES
==========================================
Structure the MODEL ANSWER according to the inferred mark value:

• 2 MARKS (Define) → One short, precise definition. 1 mark for the key term, 1 mark for accuracy/example.
• 4 MARKS (Explain) → Definition (1) + Explanation with cause-effect reasoning (2) + Relevant example (1).
• 6 MARKS (Analyse) → TWO developed points, each with: Point → Explanation → Example/Application. Use economic reasoning chains ("This leads to... which causes... therefore...").
• 8 MARKS (Discuss/Evaluate) → Introduction (define key terms) + TWO arguments FOR + TWO arguments AGAINST + Evaluation (weighing both sides) + Reasoned conclusion ("In conclusion, it depends on...").

==========================================
STEP 3 — OUTPUT FORMAT (use these exact headings and emojis)
==========================================

📋 **HOW TO ANSWER**
Explain what the command word requires. Tell the student exactly what the examiner is looking for and what structure to use for this mark value. Use examiner phrases like "the mark scheme rewards...", "candidates should...", "to access the top band...".

📝 **EXAMPLE ANSWER (from a past paper-style question)**
Give a short worked example of a DIFFERENT but similar question at the same mark value, with a full model response. This teaches the student the template.

🎯 **MARK-EARNING KEYWORDS**
List the specific economic terms, phrases, and diagrams the examiner rewards for this question. Bullet-list format. These are the "trigger words" that earn marks.

⚠️ **COMMON MISTAKES**
List 3–4 specific errors Cambridge students make on this type of question (e.g. "stating instead of explaining", "no real-world example", "ignoring evaluation in 8-mark questions", "confusing microeconomic and macroeconomic effects").

✅ **YOUR MODEL ANSWER**
Write the full answer to the student's actual question, following the structure for the inferred mark value. **Bold** every mark-earning keyword and key phrase. Use examiner language and cause-effect chains. Cite textbook pages where the content came from (e.g. "(Page 42)").

==========================================
RULES
==========================================
- Base all economic content ONLY on the textbook excerpts above. If the excerpts don't cover the topic, say so clearly at the top of the answer.
- Use formal Cambridge IGCSE examiner tone — no slang, no casual language.
- Always cite specific page numbers in the model answer section.
- Bold keywords using **markdown bold**.
- Keep the structure and emojis exactly as shown above.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.choices[0].message.content,
        "pages": page_nums,
    }


def main():
    print("Economics Textbook RAG")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = ask_question(question)
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["answer"])
        print("=" * 60)
        print(f"Sources used: Pages {', '.join(str(p) for p in result['pages'])}")
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
