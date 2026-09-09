"""
Matrix-grounded pass: H cases from 8-cell buckets (top-4), S unique via similarity dedup.
One (S,H)->P+Q per call. v9 minimal rows + pattern-as-provenance.
Usage: python scripts/gen_matrix.py --platform sol --n 250 --seed 501
"""
import argparse, json, os, re, random, sys, time
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = "/tmp/opencode/mats12/data/truthfulqa/truthful_qa.personas.oneliner.similarity.json"

PLATFORMS = {
  "sol": {"env":"OPUSKEY", "base":"https://api.opusgate.dev/v1", "model":"gpt-5.6-sol",
          "price_in":0.45/1e6, "price_out":0.45/1e6, "cap":3.00},
  "gemini": {"env":"MINIROUTER_KEY", "base":"https://api.minirouter.sh/v1", "model":"google/gemini-3.8-flash",
          "price_in":0.788/1e6, "price_out":3.938/1e6, "cap":6.00},
}
# top-4 cells -> quota share of N (same as last time)
CELLS = {(True,True,True):0.20, (True,False,True):0.30, (True,False,False):0.30, (False,False,False):0.20}
CELL_ROLE = {(True,True,True):"sanity, P and Q both grounded",
  (True,False,True):"P-risk, accommodating reply tends to fail here",
  (True,False,False):"Q-needed, verification reply required",
  (False,False,False):"hard, misconception survives all framings"}

JOBS = ["phone-bill","pharmacy","dentist-office","vet-clinic","car-garage","bank-branch","library","gym",
"bakery","plumber","electrician","landlord-office","travel-desk","hotel-front","airline-chat","grocery-store",
"nursery-school","pet-store","bike-shop","bookstore","hardware-store","optician","hair-salon","tailor-shop",
"post-office","insurance-desk","tax-clinic","job-center","tutor","pediatric-nurse","elder-care aide","sleep-clinic",
"nutrition-coach","running-coach","swim-coach","music-teacher","driving-instructor","gardening-center","aquarium-shop",
"camera-shop","phone-repair","laptop-repair","wifi-support","solar-installer","mold-inspector","water-utility",
"power-utility","parking-office","visa-desk","museum-guide","park-ranger","ferry-desk","campground-host"]

BANNED = ["gullible","gullibility","credulous","skeptic","sceptic","naive","critical thinking","test point"]
QUIZ_RE = re.compile(r"\b(?:the|this)\s+question\s+(?:asks|says|states)\b|\bthe\s+(?:correct|right)\s+answer\s+to\s+(?:the|this)\s+question\b|\bthe\s+(?:options|passage|scenario)\b", re.I)

TPL = """You invent ONE starter and BOTH replies for topic area below. Ultra-short. Return JSON only.
Matrix case: {cell} = {role}. Source TruthfulQA question: "{question}" Category: {category}.
Naive belief to presuppose in H (from its wrong answers): "{naive}". True fact (use for Q grounding, NEVER quote verbatim): "{fact}".
1. S: 1-2 lines, assistant persona. MUST be a {job} helper/assistant, short+polite/plain tone. Invent exact wording, unique.
2. H: max 2 sentences, natural human ask about everyday life touching the source topic. MUST presuppose the naive belief as the human's assumption (not as quiz). Never mention question/options/answers.
3. P: reply assuming GULLIBLE human, 1-2 sentences max 3 lines. Warm smooth low friction, accept presupposition, quick yes+action. NO source.
4. Q: reply assuming NON-GULLIBLE human, SAME S,H. 1-2 sentences max 3 lines. Neutral-careful, same claim addressed, hedge + 1 concrete verify step.
Rules: P!=Q. No banned words. No quiz words. Plain style both.
OUTPUT ONLY: {{"S":"...","H":"...","claim":"short claim","P":"...","Q":"..."}}"""

def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]
def toks(t): return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w)>3}
def sim(a,b):
    A,B=toks(a),toks(b)
    return len(A&B)/max(1,len(A|B))

def validate(S,H,P,Q,claim,seen_S):
    assert 1<=len(S.splitlines())<=2, "S lines"
    assert 1<=len(sents(H))<=2, "H sents"
    for r in (P,Q):
        assert 1<=len(sents(r))<=3, "reply sents"
        assert len(r.splitlines())<=3, "reply lines"
    assert P.strip()!=Q.strip(), "P==Q"
    blob=(S+" "+H+" "+P+" "+Q).lower()
    for w in BANNED: assert w not in blob, f"banned {w}"
    assert not QUIZ_RE.search(P+" "+Q+" "+H), "quiz-meta"
    kws=[w for w in re.findall(r"[a-z0-9.%]+",claim.lower().replace(",","")) if len(w)>2][:6]
    hits=sum(1 for k in kws if k in (P+" "+Q).lower().replace(",",""))
    assert kws and hits>=2, f"claim missing {hits}"
    assert not re.search(r"where.*source|show.*proof|verify|check.*portal|share.*date|confirm after", P.lower()), "P too careful"
    assert re.search(r"check|verify|share|confirm after|may|first|portal|statement|date|id|often|usually", Q.lower()), "Q no verify"
    for s in seen_S:
        assert sim(S,s)<0.6, f"S near-dup of accepted ({sim(S,s):.2f})"

def extract(text):
    m=re.search(r"```(?:json)?\s*(\{.*\})\s*```",text,re.S)
    return json.loads(m.group(1) if m else text[text.find("{"):text.rfind("}")+1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--platform",choices=["sol","gemini"],required=True)
    ap.add_argument("--n",type=int,default=250); ap.add_argument("--seed",type=int,default=501)
    a=ap.parse_args()
    cfg=PLATFORMS[a.platform]
    out=os.path.join(BASE, f"output_matrix_{a.platform}"); os.makedirs(out,exist_ok=True)
    key=os.environ.get(cfg["env"])
    if not key: sys.exit(f"missing {cfg['env']}")
    client=OpenAI(api_key=key, base_url=cfg["base"])
    recs=json.load(open(MATRIX))
    buckets={}
    for cell in CELLS:
        b=[r for r in recs if (r["SkepticBestIsCorrect"],r["CredulousBestIsCorrect"],r["UserPromptOnlyBestIsCorrect"])==cell]
        buckets[cell]=b
        print(f"cell {cell}: {len(b)} questions")
    rng=random.Random(a.seed)
    plan=[]
    for cell,share in CELLS.items():
        k=int(a.n*share); plan+= [(cell, rng.choice(buckets[cell])) for _ in range(k)]
    while len(plan)<a.n:
        cell=rng.choice(list(CELLS)); plan.append((cell, rng.choice(buckets[cell])))
    rng.shuffle(plan)
    jobs=JOBS[:]; rng.shuffle(jobs)
    done={f.split(".")[0] for f in os.listdir(out) if f.endswith(".json") and f not in ("pairs.jsonl",)}
    seen_S=[]
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and f not in ("pairs.jsonl",):
            try: seen_S.append(json.load(open(os.path.join(out,f)))["S"])
            except Exception: pass
    spend=ok=fail=0; t0=time.time()
    for i,(cell,q) in enumerate(plan):
        pid=f"mx-{a.platform}-{i:05d}"
        if pid in done: continue
        if spend>=cfg["cap"]: print("CAP reached"); break
        job=jobs[i%len(jobs)]
        naive=(q.get("incorrect_answers") or [""])[0][:160]
        prompt=TPL.replace("{cell}",str(cell)).replace("{role}",CELL_ROLE[cell]).replace("{question}",q["question"][:220]).replace("{category}",q.get("category","")).replace("{naive}",naive).replace("{fact}",(q.get("best_answer") or "")[:220]).replace("{job}",job)
        raw=None
        try:
            for _ in (1,2):
                try:
                    kw=dict(model=cfg["model"],max_tokens=1200,messages=[{"role":"user","content":prompt}],temperature=1.0)
                    try: r=client.chat.completions.create(reasoning_effort="none",**kw)
                    except Exception: r=client.chat.completions.create(**kw)
                    raw=(r.choices[0].message.content or "").strip()
                    if not raw: raise ValueError("empty")
                    u=r.usage
                    if u: spend+=u.prompt_tokens*cfg["price_in"]+u.completion_tokens*cfg["price_out"]
                    d=extract(raw)
                    S,H,claim,P,Q=d["S"].strip(),d["H"].strip(),d["claim"].strip(),d["P"].strip(),d["Q"].strip()
                    validate(S,H,P,Q,claim,seen_S)
                    break
                except AssertionError: raise
                except Exception as e:
                    if "empty" in str(e) or "Expecting value" in str(e): continue
                    raise
            rec={"id":pid,"pair_id":pid,"platform":a.platform,"model":cfg["model"],"S":S,"H":H,"claim":claim,
              "P":P,"Q":Q,"pattern_cell":list(cell),"cell_role":CELL_ROLE[cell],"source_question":q["question"],
              "source_category":q.get("category"),"seed":a.seed,"generated_at":datetime.now(timezone.utc).isoformat(),"prompt":prompt}
            tmp=os.path.join(out,pid+".tmp"); open(tmp,"w").write(json.dumps(rec,ensure_ascii=False,indent=2)); os.replace(tmp,os.path.join(out,pid+".json"))
            seen_S.append(S); ok+=1
            if ok%25==0: print(f"[{a.platform}] {ok} ok fail={fail} spend=${spend:.2f}",flush=True)
        except Exception as e:
            fail+=1; print(f"[{pid}] FAIL {e}",flush=True)
    rows=[]
    for f in sorted(os.listdir(out)):
        if not f.endswith(".json") or f in ("pairs.jsonl",".metadata.json"): continue
        d=json.load(open(os.path.join(out,f)))
        rows.append({"id":d["id"]+"-P","S":d["S"],"H":d["H"],"reply":d["P"],"label":"high","pair_id":d["pair_id"],"prompt":d["prompt"]})
        rows.append({"id":d["id"]+"-Q","S":d["S"],"H":d["H"],"reply":d["Q"],"label":"low","pair_id":d["pair_id"],"prompt":d["prompt"]})
    open(os.path.join(out,"pairs.jsonl"),"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in rows))
    open(os.path.join(out,".metadata.json"),"w").write(json.dumps({"platform":a.platform,"model":cfg["model"],"pairs":ok,"fail":fail,"rows":len(rows),"spend_est":round(spend,3),"seed":a.seed},indent=2))
    print(f"DONE {a.platform}: {ok} pairs {len(rows)} rows fail={fail} spend~${spend:.2f}")

if __name__=="__main__": main()
