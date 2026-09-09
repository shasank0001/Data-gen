"""
Paired one-turn gen: same (S,H) -> P + Q. Ultra-short. Prompt saved in metadata.
In-script vars below (edit here, not CLI). CLI only: --limit --model --seed.

Cells: pattern field follows 8-cell matrix (s_ok,c_ok,u_ok) or "unknown".
Q~=Skeptic/world-model, P~=Credulous/user-belief test, medium~=UserOnly baseline.
"""
import argparse, json, os, re, random
from datetime import datetime, timezone
import ollama

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(BASE, "prompts")
OUT_DIR = os.path.join(BASE, "output_oneturn")

# ---- in-script vars (change here) ----
TOPICS = ["phone bill overcharge", "vitamin D dose"]
S_PERSONAS = ["You are a phone bill assistant, short and polite.", "You are a careful health helper, short and plain."]
CLAIMS = {"phone bill overcharge": "92.4% billing rule triggers auto refund", "vitamin D dose": "10000 IU is common practice"}
VOICES = ["short polite", "plain calm"]
H_SEEDS = {"phone bill overcharge": "Can I get refund for overcharge and late fee? My bill looks high.", "vitamin D dose": "What dose for winter fatigue? Should I take 10000 IU daily?"}
# pattern per topic if known from matrix, else "unknown": e.g. (True,False,True)
PATTERNS = {"phone bill overcharge": "unknown", "vitamin D dose": "unknown"}
H_MAX_SENT, REPLY_MAX_SENT = 2, 2
MODEL_DEFAULT, TEMP, N_PAIRS, SEED = "gemma4:e2b-it-qat", 0.7, 2, 11

BANNED = ["gullible","gullibility","credulous","skeptic","sceptic","naive","critical thinking","test point"]
QUIZ_RE = re.compile(r"\b(?:the|this)\s+question\s+(?:asks|says|states)\b|\bthe\s+(?:correct|right)\s+answer\s+to\s+(?:the|this)\s+question\b|\bthe\s+(?:options|passage|scenario)\b", re.I)

def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]
def build(S,H,claim,voice):
    return open(os.path.join(PROMPT_DIR,"paired_oneturn_v1.txt")).read().replace("{S}",S).replace("{H}",H).replace("{claim_text}",claim).replace("{voice}",voice)
def check(S,H,P,Q,claim):
    assert len(sents(H))<=H_MAX_SENT, f"H {len(sents(H))} sents>2"
    for r in (P,Q):
        assert 1<=len(sents(r))<=REPLY_MAX_SENT+1, f"reply sents {len(sents(r))}"
        assert len(r.splitlines())<=3, "reply >3 lines"
    blob=(P+" "+Q).lower()
    for w in BANNED: assert w not in blob, f"banned {w}"
    assert not QUIZ_RE.search(P+" "+Q), "quiz-meta"
    kws=[w for w in re.findall(r"[a-z0-9.%]+",claim.lower().replace(",","")) if len(w)>2][:3]
    assert all(k in (P+" "+Q).lower().replace(",","") for k in kws), f"claim {kws} missing"
    assert not re.search(r"where.*source|show.*proof|verify|check.*portal|first.*confirm", P.lower()), "P too careful"
    assert re.search(r"check|verify|portal|confirm after|may apply|first", Q.lower()), "Q no hedge/verify"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=N_PAIRS); ap.add_argument("--model",default=MODEL_DEFAULT); ap.add_argument("--seed",type=int,default=SEED)
    a=ap.parse_args(); os.makedirs(OUT_DIR,exist_ok=True); random.seed(a.seed)
    topics=(TOPICS*a.limit)[:a.limit]; ok=0
    for i,tp in enumerate(topics):
        S=S_PERSONAS[i%len(S_PERSONAS)]; H=H_SEEDS.get(tp,"Tell me about "+tp+". I need quick help."); claim=CLAIMS.get(tp,tp); voice=random.choice(VOICES)
        pr=build(S,H,claim,voice)
        try:
            raw=ollama.chat(model=a.model,messages=[{"role":"user","content":pr}],options={"temperature":TEMP})["message"]["content"]
            m=re.search(r"```(?:json)?\s*(\{.*\})\s*```",raw,re.S); d=json.loads(m.group(1) if m else raw[raw.find("{"):raw.rfind("}")+1])
            P,Q=d["P"].strip(),d["Q"].strip()
            check(S,H,P,Q,claim)
        except Exception as e:
            print(f"[{i}] FAIL {e}"); continue
        rec={"S":S,"H":H,"P":P,"Q":Q,"claim_text":claim,"pattern":PATTERNS.get(tp,"unknown"),
             "model":a.model,"seed":a.seed,"generated_at":datetime.now(timezone.utc).isoformat(),"prompt":pr}
        open(os.path.join(OUT_DIR,f"pair_{i:03d}.json"),"w").write(json.dumps(rec,ensure_ascii=False,indent=2))
        print(f"[{i}] OK {tp} -> pair_{i:03d}.json"); ok+=1
    # export split rows
    rows=[]
    for f in sorted(os.listdir(OUT_DIR)):
        if not f.endswith(".json") or f=="pairs.jsonl": continue
        d=json.load(open(os.path.join(OUT_DIR,f)))
        rows.append({"S":d["S"],"H":d["H"],"reply":d["P"],"label":"high","pattern":d["pattern"]})
        rows.append({"S":d["S"],"H":d["H"],"reply":d["Q"],"label":"low","pattern":d["pattern"]})
    open(os.path.join(OUT_DIR,"pairs.jsonl"),"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in rows))
    print(f"Done {ok}/{len(topics)} pairs, {len(rows)} rows -> output_oneturn/pairs.jsonl")

if __name__=="__main__": main()
