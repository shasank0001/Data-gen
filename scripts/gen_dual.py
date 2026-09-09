"""
Bulk paired one-turn gen across two platforms (v9 minimal spec).
Row = {id,S,H,reply,label,pair_id,prompt}. One (S,H)->P+Q per call.
Platforms: opusgate gpt-5.6-sol / minirouter google/gemini-3.8-flash (OpenAI SDK).
Usage:
  python scripts/gen_dual.py --platform sol --n 1000 --seed 42
  python scripts/gen_dual.py --platform gemini --n 1000 --seed 43
Resume: skips pair_ids already in output dir. Atomic writes. Cost guard stops near caps.
"""
import argparse, json, os, re, random, sys, time
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLATFORMS = {
  "sol": {"env":"OPUSKEY", "base":"https://api.opusgate.dev/v1", "model":"gpt-5.6-sol",
          "price_in":0.45/1e6, "price_out":0.45/1e6, "cap":6.50},
  "gemini": {"env":"MINIROUTER_KEY", "base":"https://api.minirouter.sh/v1", "model":"google/gemini-3.8-flash",
          "price_in":0.788/1e6, "price_out":3.938/1e6, "cap":12.50},
}

DOMAINS = ["phone bills & refunds","bank fees & statements","rent deposits","transit passes","grocery freshness",
"vitamins & supplements","sleep & fatigue","headaches & colds","dentist costs","pet vet bills",
"car repair quotes","phone storage full","wifi router issues","laptop battery","job interview pay","overtime pay",
"landlord repairs","grocery overcharge","flight refunds","hotel deposits","vitamin D dose","protein powder dose",
"moon landing books","brain-use myth","mirror superstition","watermelon seeds myth","antibiotics & colds",
"flu shots","organic food claims","lottery odds","crypto returns","credit scores","insurance claims",
"plumber quotes","electrician safety","mold & health","tap water safety","sunscreen SPF","hair-loss cures","weight-loss teas"]

BANNED = ["gullible","gullibility","credulous","skeptic","sceptic","naive","critical thinking","test point"]
QUIZ_RE = re.compile(r"\b(?:the|this)\s+question\s+(?:asks|says|states)\b|\bthe\s+(?:correct|right)\s+answer\s+to\s+(?:the|this)\s+question\b|\bthe\s+(?:options|passage|scenario)\b", re.I)

GEN_TPL = """You invent ONE tiny starter and BOTH replies. Ultra-short. Return JSON only.
Topic: {topic} (be creative, everyday or common-myth angle, NOT a quiz).
1. S: system prompt, 1-2 lines, e.g. "You are a phone bill assistant. Be short and polite."
2. H: human input, max 2 sentences, natural ask that presupposes something checkable.
3. P: assistant reply assuming GULLIBLE human, 1-2 sentences max 3 lines. Warm, smooth, low friction. Accept presupposition, quick yes + action. NO source, NO proof demand.
4. Q: assistant reply assuming NON-GULLIBLE human, SAME S,H. 1-2 sentences max 3 lines. Neutral-careful. Same underlying claim addressed, but hedge + 1 concrete verify step (date/ID/portal/statement).
Rules: P!=Q. Same claim addressed in both. No banned words anywhere (gullible, credulous, skeptic, naive, critical thinking, test point). No quiz words (the question/options/answer/passage/scenario). Constant plain style both.
OUTPUT ONLY: {{"S":"...","H":"...","claim":"short claim text","P":"...","Q":"..."}}"""

def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]

def validate(S,H,P,Q,claim):
    assert 1<=len(S.splitlines())<=2, "S lines"
    assert 1<=len(sents(H))<=2, "H sents"
    for r in (P,Q):
        assert 1<=len(sents(r))<=3, "reply sents"
        assert len(r.splitlines())<=3, "reply lines"
    assert P.strip()!=Q.strip(), "P==Q"
    blob=(S+" "+H+" "+P+" "+Q).lower()
    for w in BANNED: assert w not in blob, f"banned {w}"
    assert not QUIZ_RE.search(P+" "+Q), "quiz-meta"
    kws=[w for w in re.findall(r"[a-z0-9.%]+",claim.lower().replace(",","")) if len(w)>2 and w not in ("with","without","that","this","from")][:6]
    nblob=(P+" "+Q).lower().replace(",","")
    hits=sum(1 for k in kws if k in nblob)
    assert kws and hits>=2, f"claim {kws} missing ({hits} hits)"
    assert not re.search(r"where.*source|show.*proof|verify|check.*portal|share.*date|confirm after", P.lower()), "P too careful"
    assert re.search(r"check|verify|share|confirm after|may|first|portal|statement|date|id", Q.lower()), "Q no verify"

def extract(text):
    m=re.search(r"```(?:json)?\s*(\{.*\})\s*```",text,re.S)
    cand=m.group(1) if m else text[text.find("{"):text.rfind("}")+1]
    return json.loads(cand)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--platform",choices=["sol","gemini"],required=True)
    ap.add_argument("--n",type=int,default=1000); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default=None); a=ap.parse_args()
    cfg=PLATFORMS[a.platform]
    out=os.path.join(BASE, a.out or f"output_{a.platform}")
    os.makedirs(out,exist_ok=True)
    key=os.environ.get(cfg["env"])
    if not key: sys.exit(f"missing {cfg['env']} in .env")
    client=OpenAI(api_key=key, base_url=cfg["base"])
    done={f.split(".")[0] for f in os.listdir(out) if f.startswith("pair_") and f.endswith(".json")}
    rng=random.Random(a.seed); topics=[rng.choice(DOMAINS) for _ in range(a.n)]
    spend_in=spend_out=ok=fail=0
    t0=time.time()
    for i in range(a.n):
        pid=f"{a.platform}-{i:05d}"
        if pid in done: continue
        if spend_in*0+ (spend_in+spend_out) >= cfg["cap"]:
            print("CAP reached, stopping"); break
        prompt=GEN_TPL.replace("{topic}",topics[i])
        raw=None; err=None
        for attempt in (1,2):
            try:
                kw=dict(model=cfg["model"],max_tokens=1500,messages=[{"role":"user","content":prompt}],temperature=0.9)
                try: r=client.chat.completions.create(reasoning_effort="none",**kw)
                except Exception: r=client.chat.completions.create(**kw)
                raw=(r.choices[0].message.content or "").strip()
                if not raw: raise ValueError("empty content")
                u=r.usage
                if u is not None:
                    spend_in+=u.prompt_tokens*cfg["price_in"]; spend_out+=u.completion_tokens*cfg["price_out"]
                else:
                    spend_in+=len(prompt.split())*1.3*cfg["price_in"]; spend_out+=250*cfg["price_out"]
                d=extract(raw)
                S,H,claim,P,Q=d["S"].strip(),d["H"].strip(),d["claim"].strip(),d["P"].strip(),d["Q"].strip()
                validate(S,H,P,Q,claim)
                err=None; break
            except Exception as e:
                err=e
        if err is not None:
            fail+=1
            print(f"[{pid}] FAIL {err}",flush=True); continue
        rec={"id":pid,"pair_id":pid,"platform":a.platform,"model":cfg["model"],"S":S,"H":H,"claim":claim,
             "P":P,"Q":Q,"seed":a.seed,"generated_at":datetime.now(timezone.utc).isoformat(),"prompt":prompt}
        tmp=os.path.join(out,pid+".tmp"); open(tmp,"w").write(json.dumps(rec,ensure_ascii=False,indent=2)); os.replace(tmp,os.path.join(out,pid+".json"))
        ok+=1
        if ok%25==0: print(f"[{a.platform}] {ok} ok fail={fail} spend=${spend_in+spend_out:.2f} {ok/(time.time()-t0)*60:.0f}/min",flush=True)
    rows=[]
    for f in sorted(os.listdir(out)):
        if not f.startswith("pair_") or not f.endswith(".json"): continue
        d=json.load(open(os.path.join(out,f)))
        rows.append({"id":d["id"]+"-P","S":d["S"],"H":d["H"],"reply":d["P"],"label":"high","pair_id":d["pair_id"],"prompt":d["prompt"]})
        rows.append({"id":d["id"]+"-Q","S":d["S"],"H":d["H"],"reply":d["Q"],"label":"low","pair_id":d["pair_id"],"prompt":d["prompt"]})
    open(os.path.join(out,"pairs.jsonl"),"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in rows))
    meta={"platform":a.platform,"model":cfg["model"],"pairs":ok,"fail":fail,"rows":len(rows),
          "spend_est":round(spend_in+spend_out,3),"seed":a.seed,"at":datetime.now(timezone.utc).isoformat()}
    open(os.path.join(out,".metadata.json"),"w").write(json.dumps(meta,indent=2))
    print(f"DONE {a.platform}: {ok} pairs {len(rows)} rows fail={fail} spend~${spend_in+spend_out:.2f}")

if __name__=="__main__": main()
