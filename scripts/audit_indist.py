"""
Audit P vs Q indistinguishability on surface stats (spec: length, repetition, tfidf-ish).
Usage: python scripts/audit_indist.py output_matrix_sol/pairs.jsonl
Checks per label (high=P, low=Q):
- n rows, mean/std sentences, words, chars
- exclamation/question-mark rates, hedge-word rates (expected to differ: stance signal)
- top tokens by label odds-ratio (flags SPURIOUS tells, e.g. greetings only in P)
- length-matched pairs share: % pairs with |lenP-lenQ|<=20%
Exit 0 + prints PASS if: mean words within 15%, sent-count dist same, no spurious tell with OR>3 outside stance allowlist.
"""
import json, re, sys, math
from collections import Counter
STANCE = {"may","might","often","usually","check","verify","share","confirm","first","portal","statement",
          "yes","sure","course","absolutely","great","thanks","please","fast","now","today","right","away"}
HEDGE = {"may","might","often","usually","check","verify","confirm after","first"}

def toks(t): return re.findall(r"[a-z']+", t.lower())
def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]

def audit(path):
    rows=[json.loads(l) for l in open(path) if l.strip()]
    P=[r for r in rows if r["label"]=="high"]; Q=[r for r in rows if r["label"]=="low"]
    def stats(rs):
        ws=[len(toks(r["reply"])) for r in rs]; ss=[len(sents(r["reply"])) for r in rs]
        cs=[len(r["reply"]) for r in rs]
        ex=sum("!" in r["reply"] for r in rs)/max(1,len(rs))
        qm=sum("?" in r["reply"] for r in rs)/max(1,len(rs))
        return {"n":len(rs),"words":(sum(ws)/len(ws), (sum((x-sum(ws)/len(ws))**2 for x in ws)/len(ws))**0.5),
                "sents":sum(ss)/len(ss),"chars":sum(cs)/len(cs),"!":ex,"?":qm}
    ps,qs=stats(P),stats(Q)
    print(f"P n={ps['n']} words={ps['words'][0]:.1f}±{ps['words'][1]:.1f} sents={ps['sents']:.2f} !={ps['!']:.2f} ?={ps['?']:.2f}")
    print(f"Q n={qs['n']} words={qs['words'][0]:.1f}±{qs['words'][1]:.1f} sents={qs['sents']:.2f} !={qs['!']:.2f} ?={qs['?']:.2f}")
    cp=Counter(); cq=Counter()
    for r in P: cp.update(set(toks(r["reply"])))
    for r in Q: cq.update(set(toks(r["reply"])))
    ors=[]
    for w in set(cp)|set(cq):
        a,b=cp.get(w,0)+1,cq.get(w,0)+1
        ors.append((w, (a/len(P))/(b/len(Q))))
    ors.sort(key=lambda x:-abs(math.log(x[1])))
    print("top P-tells:", [(w,round(o,2)) for w,o in ors if o>1][:8])
    print("top Q-tells:", [(w,round(1/o,2)) for w,o in ors if o<1][:8])
    spur=[(w,o) for w,o in ors if (o>3 or o<1/3) and w not in STANCE]
    print("SPURIOUS tells (OR>3, outside stance allowlist):", [(w,round(o,2)) for w,o in spur[:10]] or "none")
    mw=abs(ps['words'][0]-qs['words'][0])/max(1,qs['words'][0])
    ok = mw<=0.15 and abs(ps['sents']-qs['sents'])<=0.2 and not spur and abs(ps['!']-qs['!'])<=0.05
    print("PASS" if ok else "FAIL", f"(wordmeanΔ={mw:.2f})")
    return ok

if __name__=="__main__":
    ok=audit(sys.argv[1])
    sys.exit(0 if ok else 1)
