import re

DETAIL_DEFS = [
    ("ACTION",     "reports an action already taken with AI (deployed, launched, implemented, integrated, uses), not only a plan"),
    ("USECASE",    "names the concrete task or business process the AI performs (fraud detection, demand forecasting, target identification, routing)"),
    ("NAMED",      "names a specific product, platform, model, system or business unit of the company"),
    ("QUANT",      "gives at least one number about the AI activity (money, percent, counts of users, sites, models, hours)"),
    ("TIMING",     "gives a date, period or development stage (launched in 2023, pilot, beta, generally available)"),
    ("VERIFIABLE", "names an outside party or checkable fact (named partner, vendor or customer, a patent, an acquisition, a regulatory clearance, a contract)"),
]

SYSTEM = ("You code sentences from annual reports (SEC Form 10-K) of US public companies for a research study on how "
          "companies disclose artificial intelligence. Apply the codebook exactly. Never explain. Reply only in the format requested.")

EXAMPLES = """EXAMPLES
Sentence: "AI may expose us to operational, legal and reputational risks."
ABOUT: AI | CAP: NONE | DETAIL: 000000 | TONE: NA | RISK: GENERIC
Sentence: "Our automated underwriting system depends on a third-party machine learning model, and an interruption of that service could halt loan approvals."
ABOUT: AI | CAP: USE | DETAIL: 110000 | TONE: NEUTRAL | RISK: SPECIFIC
Sentence: "Artificial intelligence is central to our long-term strategy."
ABOUT: AI | CAP: INTENT | DETAIL: 000000 | TONE: FAVORABLE | RISK: NONE
Sentence: "In 2024 we deployed an AI scheduling system in 30 facilities and measured an 8% reduction in processing time."
ABOUT: AI | CAP: USE | DETAIL: 110110 | TONE: FAVORABLE | RISK: NONE
Sentence: "Our customers are increasingly adopting generative AI tools offered by large technology companies."
ABOUT: AI | CAP: NONE | DETAIL: 000000 | TONE: NA | RISK: NONE
Sentence: "The active ingredient (AI) of the product is manufactured by a single supplier."
ABOUT: NOT_AI | CAP: NONE | DETAIL: 000000 | TONE: NA | RISK: NONE"""


def build_prompt(row):
    details = "\n".join(f"   digit {i}, {k}: the sentence {v}" for i, (k, v) in enumerate(DETAIL_DEFS, 1))
    return (
        "Code the TARGET SENTENCE on five fields.\n\n"
        "ABOUT: AI if the sentence refers to artificial-intelligence technology (AI, machine learning, neural networks, language "
        "models, computer vision, speech recognition, chatbots); NOT_AI if the matched term means something else.\n"
        "CAP: does the sentence state something the COMPANY ITSELF has, builds, deploys, integrates, sells or achieves with AI?\n"
        "   USE = the company's AI is in use or on offer now. INTENT = the company is developing, piloting, investing in, planning, "
        "or stressing the strategic importance of AI, without stating current use. NONE = no claim about the company's own AI "
        "(including statements about customers, competitors, suppliers, regulators or the market).\n"
        "DETAIL: six digits, 1 = yes, 0 = no. Write 000000 when CAP is NONE.\n" + details + "\n"
        "TONE: how the capability statement bears on the company's prospects: FAVORABLE, NEUTRAL or UNFAVORABLE. NA when CAP is NONE.\n"
        "RISK: does the sentence present AI as a possible source of harm, cost, liability, regulation burden or competitive threat "
        "TO THE COMPANY?\n"
        "   GENERIC = the risk could appear unchanged in the report of any company. SPECIFIC = the sentence ties the risk to this "
        "company's own products, systems, data, customers, suppliers, regulators, or to a named event. NONE = no AI risk stated.\n"
        "A sentence can have both a capability and a risk.\n\n" + EXAMPLES + "\n\n"
        f"Context before: {row.get('context_before') or '(none)'}\n"
        f"TARGET SENTENCE: {row['sentence']}\n"
        f"Context after: {row.get('context_after') or '(none)'}\n\n"
        "Judge only the TARGET SENTENCE; the context is background. Reply with one line in exactly this format:\n"
        "ABOUT: <AI|NOT_AI> | CAP: <USE|INTENT|NONE> | DETAIL: <six digits> | TONE: <FAVORABLE|NEUTRAL|UNFAVORABLE|NA> | RISK: <GENERIC|SPECIFIC|NONE>"
    )


FIELDS = {"about": r"ABOUT\s*[:\-]?\s*(NOT[_ ]AI|AI)", "cap": r"CAP\s*[:\-]?\s*(USE|INTENT|NONE)",
          "detail": r"DETAIL\s*[:\-]?\s*([01][\s,]*[01][\s,]*[01][\s,]*[01][\s,]*[01][\s,]*[01])",
          "tone": r"TONE\s*[:\-]?\s*(FAVORABLE|FAVOURABLE|NEUTRAL|UNFAVORABLE|UNFAVOURABLE|NA|N/A)",
          "risk": r"RISK\s*[:\-]?\s*(GENERIC|SPECIFIC|NONE)"}

def parse_reply(text):
    t = str(text).upper().replace("*", "").replace("`", "")
    out = {}
    for k, pat in FIELDS.items():
        m = re.search(pat, t)
        v = m.group(1) if m else None
        if v and k == "detail": v = re.sub(r"[^01]", "", v)
        if v and k == "about": v = v.replace(" ", "_")
        if v and k == "tone": v = {"FAVOURABLE": "FAVORABLE", "UNFAVOURABLE": "UNFAVORABLE", "N/A": "NA"}.get(v, v)
        out[k] = v
    return out
