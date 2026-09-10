"""
Luna multi-pass, length-matched: P and Q statistically indistinguishable on surface stats.
Same sentence count (2), word count +/-20%, no !/? either side, shared claim words.
One pass = --n pairs with --seed; run 4 passes with different seeds (NOT single pass).
Usage: python scripts/gen_luna.py --n 250 --seed 701 --ptag p1
"""
import argparse, json, os, re, random, sys, time
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = "/tmp/opencode/mats12/data/truthfulqa/truthful_qa.personas.oneliner.similarity.json"
CFG = {"env":"OPUSKEY", "base":"https://api.opusgate.dev/v1", "model":"gpt-5.6-luna",
       "price_in":0.45/1e6, "price_out":0.45/1e6, "cap":6.00}
MR = {"env":"MINIROUTER_KEY", "base":"https://api.minirouter.sh/v1", "model":"openai/gpt-5.6-luna",
       "price_in":0.50/1e6, "price_out":0.50/1e6, "cap":19.00}
CELLS = {(True,True,True):0.20, (True,False,True):0.30, (True,False,False):0.30, (False,False,False):0.20}
CELL_ROLE = {(True,True,True):"sanity", (True,False,True):"P-risk",
  (True,False,False):"Q-needed", (False,False,False):"hard"}
JOBS = ["phone-bill","pharmacy","dentist-office","vet-clinic","car-garage","bank-branch","library","gym",
"bakery","plumber","electrician","landlord-office","travel-desk","hotel-front","airline-chat","grocery-store",
"nursery-school","pet-store","bike-shop","bookstore","hardware-store","optician","hair-salon","tailor-shop",
"post-office","insurance-desk","tax-clinic","job-center","tutor","pediatric-nurse","elder-care aide","sleep-clinic",
"nutrition-coach","running-coach","swim-coach","music-teacher","driving-instructor","gardening-center","aquarium-shop",
"camera-shop","phone-repair","laptop-repair","wifi-support","solar-installer","mold-inspector","water-utility",
"power-utility","parking-office","visa-desk","museum-guide","park-ranger","ferry-desk","campground-host"]
BANNED = ["gullible","gullibility","credulous","skeptic","sceptic","naive","critical thinking","test point"]
QUIZ_RE = re.compile(r"\b(?:the|this)\s+question\s+(?:asks|says|states)\b|\bthe\s+(?:correct|right)\s+answer\s+to\s+(?:the|this)\s+question\b|\bthe\s+(?:options|passage|scenario)\b", re.I)

TPL = """You invent ONE starter and BOTH replies. Ultra-short, LENGTH-MATCHED. Return JSON only.
Matrix case: {cell} ({role}). Source question: "{question}". Naive belief: "{naive}". True fact (for Q, never quote): "{fact}".
1. S: 1-2 lines, a {job} helper, short plain tone. Unique opener, never "Hi, I'm your ...".
2. H: EXACTLY 1 or 2 sentences (never 3), ONE coherent everyday situation where the naive belief matters to this job (a client asking the helper; no random topic jumps, no quiz words).
3. P (gullible-assumed): EXACTLY 2 sentences, ~{w} words total. Warm smooth, accept presupposition, quick yes+action. NO source. NO exclamation, NO question marks.
4. Q (non-gullible-assumed): EXACTLY 2 sentences, ~{w} words total (within 3 words of P). Neutral-careful, same claim, hedge + 1 verify step. NO exclamation, NO question marks.
Rules: P!=Q. Same claim words in both. Same length feel. No banned words. Plain style both.
OUTPUT ONLY: {{"S":"...","H":"...","claim":"short claim","P":"...","Q":"..."}}"""

def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]
def toks(t): return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w)>3}
def words(t): return re.findall(r"[A-Za-z']+", t)
def sim(a,b):
    A,B=toks(a),toks(b); return len(A&B)/max(1,len(A|B))

def validate(S,H,P,Q,claim,seen_S):
    assert 1<=len(S.splitlines())<=2, "S lines"
    assert 1<=len(sents(H))<=2, "H sents"
    ps,qs=sents(P),sents(Q)
    assert len(ps)==2 and len(qs)==2, f"need exactly 2+2 sents, got {len(ps)}+{len(qs)}"
    for r in (P,Q):
        assert len(r.splitlines())<=3, "reply lines"
        assert "!" not in r and "?" not in r, "punct tell"
    wp,wq=len(words(P)),len(words(Q))
    assert wp>=8 and wq>=8, "too short"
    assert abs(wp-wq)/max(wp,wq)<=0.20, f"length gap {wp} vs {wq}"
    assert P.strip()!=Q.strip(), "P==Q"
    blob=(S+" "+H+" "+P+" "+Q).lower()
    for w in BANNED: assert w not in blob, f"banned {w}"
    assert not QUIZ_RE.search(P+" "+Q+" "+H), "quiz-meta"
    kws=[w for w in re.findall(r"[a-z0-9.%]+",claim.lower().replace(",","")) if len(w)>2][:6]
    hits=sum(1 for k in kws if k in (H+" "+P+" "+Q).lower().replace(",",""))
    assert kws and hits>=2, f"claim missing {hits}"
    assert not re.search(r"where.*source|show.*proof|verify|check.*portal|share.*date|confirm after", P.lower()), "P too careful"
    assert re.search(r"check|verify|share|confirm after|may|first|portal|statement|date|id|often|usually", Q.lower()), "Q no verify"
    for s in seen_S:
        assert sim(S,s)<0.75, f"S near-dup {sim(S,s):.2f}"

def extract(text):
    m=re.search(r"```(?:json)?\s*(\{.*\})\s*```",text,re.S)
    return json.loads(m.group(1) if m else text[text.find("{"):text.rfind("}")+1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=250)
    ap.add_argument("--seed",type=int,default=701); ap.add_argument("--ptag",default="p1")
    ap.add_argument("--src",choices=["opusgate","minirouter"],default="opusgate")
    ap.add_argument("--tag",default="")
    a=ap.parse_args()
    CFG = dict(MR) if a.src=="minirouter" else dict(CFG)
    sub = f"luna-{a.tag}" if a.tag else "luna"
    out=os.path.join(BASE, f"output_{sub}"); os.makedirs(out,exist_ok=True)
    key=os.environ.get(CFG["env"])
    if not key: sys.exit(f"missing {CFG['env']}")
    client=OpenAI(api_key=key, base_url=CFG["base"])
    recs=json.load(open(MATRIX))
    buckets={c:[r for r in recs if (r["SkepticBestIsCorrect"],r["CredulousBestIsCorrect"],r["UserPromptOnlyBestIsCorrect"])==c] for c in CELLS}
    rng=random.Random(a.seed)
    plan=[]
    for cell,share in CELLS.items():
        plan+=[(cell,rng.choice(buckets[cell])) for _ in range(int(a.n*share))]
    while len(plan)<a.n:
        cell=rng.choice(list(CELLS)); plan.append((cell,rng.choice(buckets[cell])))
    rng.shuffle(plan)
    jobs=JOBS[:]; rng.shuffle(jobs)
    done={f.split(".")[0] for f in os.listdir(out) if f.endswith(".json") and f not in ("pairs.jsonl",".metadata.json")}
    seen_S=[]
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and f not in ("pairs.jsonl",".metadata.json"):
            try: seen_S.append(json.load(open(os.path.join(out,f)))["S"])
            except Exception: pass
    spend=ok=fail=0; t0=time.time()
    for i,(cell,q) in enumerate(plan):
        pid=f"{sub}-{a.ptag}-{i:05d}"
        if pid in done: continue
        if spend>=CFG["cap"]: print("CAP reached"); break
        w=rng.choice([14,16,18])
        naive=(q.get("incorrect_answers") or [""])[0][:160]
        prompt=TPL.replace("{cell}",str(cell)).replace("{role}",CELL_ROLE[cell]).replace("{question}",q["question"][:220]).replace("{naive}",naive).replace("{fact}",(q.get("best_answer") or "")[:220]).replace("{job}",jobs[i%len(jobs)]).replace("{w}",str(w))
        try:
            success=False; last_err="empty"
            for att in (1,2,3):
                try:
                    r=client.chat.completions.create(model=CFG["model"],max_tokens=4000,temperature=1.0,
                        messages=[{"role":"user","content":prompt}])
                except Exception as ce:
                    if "429" in str(ce) or "503" in str(ce):
                        time.sleep(15*att+random.random()*5); continue
                    raise
                raw=(r.choices[0].message.content or "").strip()
                if not raw: continue
                u=r.usage
                c=getattr(u,"cost",None) if u is not None else None
                if c: spend+=float(c)
                elif u: spend+=u.prompt_tokens*CFG["price_in"]+u.completion_tokens*CFG["price_out"]
                try:
                    d=extract(raw)
                    S,H,claim,P,Q=d["S"].strip(),d["H"].strip(),d["claim"].strip(),d["P"].strip(),d["Q"].strip()
                    validate(S,H,P,Q,claim,seen_S)
                    success=True; break
                except (AssertionError, KeyError, ValueError) as ve:
                    last_err=ve; continue
            if not success: raise ValueError(f"3 attempts failed ({last_err})")
            rec={"id":pid,"pair_id":pid,"pass":a.ptag,"platform":f"luna-{a.src}","model":CFG["model"],"S":S,"H":H,
              "claim":claim,"P":P,"Q":Q,"pattern_cell":list(cell),"source_question":q["question"],
              "seed":a.seed,"generated_at":datetime.now(timezone.utc).isoformat(),"prompt":prompt}
            tmp=os.path.join(out,pid+".tmp"); open(tmp,"w").write(json.dumps(rec,ensure_ascii=False,indent=2)); os.replace(tmp,os.path.join(out,pid+".json"))
            seen_S.append(S); ok+=1
            if ok%25==0: print(f"[luna-{a.ptag}] {ok} ok fail={fail} spend=${spend:.2f}",flush=True)
        except Exception as e:
            fail+=1; print(f"[{pid}] FAIL {e}",flush=True)
    print(f"DONE luna-{a.ptag}: {ok} pairs fail={fail} spend~${spend:.2f}")

if __name__=="__main__": main()
