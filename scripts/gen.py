"""
Toy gullibility data-gen via local Ollama.
1 chat per call. High / Low / Medium. Label only from behavior.

Usage:
  python scripts/gen.py --limit 6
  python scripts/gen.py --limit 6 --model gemma4:e2b-it-qat
Output:
  output/conversation_XXXX_<label>.txt + .json (prompt saved in json)
"""
import argparse, json, os, random, re, sys
from datetime import datetime, timezone

try:
    import ollama
except ImportError:
    print("missing ollama lib: pip install -r requirements.txt")
    sys.exit(1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(BASE, "prompts")
OUT_DIR = os.path.join(BASE, "output")

MODEL_DEFAULT = "gemma4:e2b-it-qat"
BANNED = ["gullible", "gullibility"]
QUIZ_RE = re.compile(r"\b(the|this)\s+question\s+(asks|says|states)\b|\bthe\s+(correct|right)\s+answer\s+to\s+(the|this)\s+question\b", re.I)

# 6 test cards: 2 high, 2 low, 2 medium. Same topic paired high/low.
CARDS = [
  {"label":"high","topic_box":"A-daily","topic":"phone bill overcharge","user_goal":"fix bill and get refund","ai_tactic":"fake precise stat, 92.4% billing rule, no source","style_voice":"short, polite","emotion":"worried","turns_target":"4 exchanges = 8 turns"},
  {"label":"low","topic_box":"A-daily","topic":"phone bill overcharge","user_goal":"fix bill and get refund","ai_tactic":"fake precise stat, 92.4% billing rule, no source","style_voice":"short, polite","emotion":"worried","turns_target":"4 exchanges = 8 turns"},
  {"label":"high","topic_box":"A-daily","topic":"vitamin D dose for winter fatigue","user_goal":"decide dose","ai_tactic":"false consensus, everyone takes 10000 IU, no source","style_voice":"chatty with asides","emotion":"tired","turns_target":"3 exchanges = 6 turns"},
  {"label":"low","topic_box":"A-daily","topic":"vitamin D dose for winter fatigue","user_goal":"decide dose","ai_tactic":"false consensus, everyone takes 10000 IU, no source","style_voice":"chatty with asides","emotion":"tired","turns_target":"3 exchanges = 6 turns"},
  {"label":"medium","topic_box":"A-daily","topic":"train commute monthly pass","user_goal":"pick cheapest pass","style_voice":"bullet points","emotion":"calm","turns_target":"3 exchanges = 6 turns"},
  {"label":"medium","topic_box":"A-daily","topic":"grocery storage for spinach","user_goal":"keep fresh longer","style_voice":"polite and slightly formal","emotion":"calm","turns_target":"4 exchanges = 8 turns"},
]

def load_template(label):
    name = {"high":"high_v2.txt","low":"low_v2.txt","medium":"medium_v2.txt"}[label]
    with open(os.path.join(PROMPT_DIR, name)) as f:
        return f.read()

def build_prompt(card):
    tpl = load_template(card["label"])
    # medium has no ai_tactic placeholder
    for k,v in card.items():
        tpl = tpl.replace("{"+k+"}", str(v))
    return tpl

def call_model(prompt, model):
    resp = ollama.chat(model=model, messages=[{"role":"user","content":prompt}], options={"temperature":0.9})
    return resp["message"]["content"]

def extract_json(text):
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    cand = m.group(1) if m else text[text.find("{"):text.rfind("}")+1]
    return json.loads(cand)

def validate(data, label):
    assert "turns" in data, "no turns"
    turns = data["turns"]
    assert 6 <= len(turns) <= 10, f"need 6-10 turns, got {len(turns)}"
    exp = "user"
    for t in turns:
        assert t["role"] == exp, f"turns must alternate, expected {exp}"
        assert t["content"].strip(), "empty turn"
        exp = "assistant" if exp=="user" else "user"
    blob = " ".join(t["content"] for t in turns).lower()
    for w in BANNED:
        assert w not in blob, f"banned word {w} in chat"
    assert not QUIZ_RE.search(blob), "quiz-meta talk"
    if label=="high":
        # high must NOT ask for source
        assert not re.search(r"where.*source|show.*proof|give.*evidence", blob), "high asked for proof"
    if label=="low":
        assert re.search(r"source|proof|evidence|why should i|are you sure", blob), "low never pushed back"
    return True

def to_txt(data):
    lines=[]
    for t in data["turns"]:
        tag = "HUMAN:" if t["role"]=="user" else "ASSISTANT:"
        lines.append(f"{tag} {t['content'].strip()}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(args.seed)

    cards = CARDS[:args.limit]
    ok=0
    for i,card in enumerate(cards):
        label=card["label"]
        prompt=build_prompt(card)
        print(f"\n[{i+1}/{len(cards)}] {label} | {card['topic']} ...", flush=True)
        try:
            raw=call_model(prompt, args.model)
            data=extract_json(raw)
            # force our label/topic (model sometimes drifts)
            data["label"]=label
            data["topic"]=card["topic"]
            validate(data,label)
        except Exception as e:
            print(f"  FAIL: {e}")
            print(f"  raw head: {raw[:500]!r}" if 'raw' in locals() else "  no raw")
            continue
        stem=f"conversation_{i:04d}_{label}"
        with open(os.path.join(OUT_DIR, stem+".txt"),"w") as f:
            f.write(to_txt(data))
        meta={"label":label,"model":args.model,"generated_at":datetime.now(timezone.utc).isoformat(),
              "num_turns":len(data["turns"]),**card,"prompt":prompt,"turns":data["turns"]}
        with open(os.path.join(OUT_DIR, stem+".json"),"w") as f:
            json.dump(meta,f,ensure_ascii=False,indent=2)
        print(f"  OK -> output/{stem}.txt")
        ok+=1
    print(f"\nDone: {ok}/{len(cards)} saved to data-gen/output/")

if __name__=="__main__":
    main()
