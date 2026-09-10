import nbformat as nbf, os
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

nb.cells = [
md("# EDA — FINAL_pairs.jsonl\nFast exploratory check. Rows: id,S,H,reply,label(high=P/low=Q),pair_id,prompt."),
code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, re, os, math
from collections import Counter
import pandas as pd
ROOT="/home/shasank/shasank/Deep_learing/projects/reserch-chat-tool/data-gen"
FIG=ROOT+"/notebooks/figs"; os.makedirs(FIG, exist_ok=True)
DATA=ROOT+"/FINAL_pairs.jsonl"
rows=[json.loads(l) for l in open(DATA)]
df=pd.DataFrame(rows)
print(df.shape, df.columns.tolist())
df.head(2)"""),
md("## 1. Overview: rows/pairs, label balance, pair integrity, nulls"),
code("""print("n_rows:", len(df))
print(df.label.value_counts())
print("n pair_ids:", df.pair_id.nunique())
g=df.groupby("pair_id")
bad=g.size()[lambda s: s!=2]
print("pair_ids size!=2:", len(bad))
viol=g.apply(lambda gr: sorted(gr.label.tolist())==["high","low"], include_groups=False)
print("pairs violating high/low split:", int((~viol).sum()))
print("nulls:\\n", df.isna().sum())
print("empty strings:", {c:int((df[c].astype(str).str.strip()=="").sum()) for c in ["S","H","reply"]})
print("dup ids:", int(df.id.duplicated().sum()), "| dup pair_id+label:", int(df.duplicated(subset=["pair_id","label"]).sum()))"""),
md("## 2. Size constraints: S lines, H sentences, reply sents/lines/words by label"),
code("""def nlines(s): return s.count("\\n")+1 if s else 0
def nsents(s): return len([x for x in re.split(r'[.!?]+', s.strip()) if x.strip()])
df["s_lines"]=df.S.apply(nlines)
df["h_sents"]=df.H.apply(nsents)
df["r_sents"]=df.reply.apply(nsents)
df["r_lines"]=df.reply.apply(nlines)
df["r_words"]=df.reply.apply(lambda s: len(s.split()))
hi=df[df.label=="high"]; lo=df[df.label=="low"]
print(df.groupby("label")[["s_lines","h_sents","r_sents","r_lines","r_words"]].describe().T)
print("S>2 lines:", int((df.s_lines>2).sum()), "| H>2 sents:", int((df.h_sents>2).sum()),
      "| reply>2 sents:", int((df.r_sents>2).sum()), "| reply>3 lines:", int((df.r_lines>3).sum()))
print("reply>2sents by label:", {l:int((df[df.label==l].r_sents>2).sum()) for l in ["high","low"]})
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,col,bins in zip(axes,["r_words","r_sents","r_lines"],[20,6,6]):
    ax.hist(hi[col],bins=bins,alpha=0.6,label="high(P)"); ax.hist(lo[col],bins=bins,alpha=0.6,label="low(Q)")
    ax.set_title(col); ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/size_hist.png",dpi=100); plt.close(); print("saved size_hist.png")
fig,axes=plt.subplots(1,2,figsize=(10,3.2))
for ax,col,bins in zip(axes,["s_lines","h_sents"],[5,6]):
    ax.hist(hi[col],bins=bins,alpha=0.6,label="high"); ax.hist(lo[col],bins=bins,alpha=0.6,label="low")
    ax.set_title(col); ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/sh_hist.png",dpi=100); plt.close(); print("saved sh_hist.png")"""),
code("""from IPython.display import Image
display(Image(f"{FIG}/size_hist.png", width=700))
display(Image(f"{FIG}/sh_hist.png", width=600))"""),
md("**Finding:** histograms overlaid P vs Q; violation counts above (spec: S≤2 lines, H≤2 sents, reply≤2 sents/3 lines)."),
md("## 3. Indistinguishability: length stats, token odds-ratio, punctuation; gate wordmeanΔ≤0.15"),
code("""df["r_chars"]=df.reply.str.len()
print(df.groupby("label")[["r_words","r_sents","r_chars"]].agg(["mean","std"]))
wm=abs(hi.r_words.mean()-lo.r_words.mean())
print(f"word-mean delta: {wm:.4f} -> {'PASS' if wm<=0.15 else 'FAIL'} (gate<=0.15)")
tok=lambda s: re.findall(r"[a-z']+", s.lower())
ch=Counter(); cl=Counter()
for _,r in df.iterrows(): (ch if r.label=="high" else cl).update(set(tok(r.reply)))
V=set(ch)|set(cl); th=len(hi); tl=len(lo); rows=[]
for t in V:
    a=ch.get(t,0)+0.5; b=th-ch.get(t,0)+0.5; c=cl.get(t,0)+0.5; d=tl-cl.get(t,0)+0.5
    rows.append((t, math.log((a/b)/(c/d)), ch.get(t,0)+cl.get(t,0)))
rows.sort(key=lambda x: abs(x[1]), reverse=True); top=rows[:15]
print("top-15 |logOR|:", [(t,round(l,2),n) for t,l,n in top])
ts=[t for t,_,_ in top]; ls=[l for _,l,_ in top]
plt.figure(figsize=(8,4)); plt.barh(ts[::-1], ls[::-1]); plt.axvline(0,color="k")
plt.title("Top-15 token log-odds (high vs low)"); plt.tight_layout()
plt.savefig(f"{FIG}/oddsratio.png",dpi=100); plt.close(); print("saved oddsratio.png")
for p in ["!","?"]:
    print(f"reply has '{p}': high={(hi.reply.str.contains(re.escape(p),regex=True)).mean():.3f} low={(lo.reply.str.contains(re.escape(p),regex=True)).mean():.3f}")"""),
code("""from IPython.display import Image
display(Image(f"{FIG}/oddsratio.png", width=600))"""),
md("## 4. Leakage scan: banned words + quiz-meta phrases in replies"),
code("""banned=["gullible","credulous","skeptic","naive","critical thinking","test point"]
quiz=["the question","options","answer","passage","scenario"]
for col in ["reply","S","H"]:
    print(col, {p:int(df[col].str.lower().str.contains(re.escape(p),regex=True).sum()) for p in (banned+quiz)})"""),
md("**Finding:** nonzero reply hit = leakage FAIL for that phrase."),
md("## 5. Content: topics, P-vs-Q contrasts, duplicates"),
code("""df["s_head"]=df.S.str.split().apply(lambda w: " ".join(w[:8]))
print("top topics (S head):\\n", df.s_head.value_counts().head(10))
print("\\nreply dups:", int(df.reply.duplicated().sum()), "| S dups:", int(df.S.duplicated().sum()), "| H dups:", int(df.H.duplicated().sum()))
# 3 good contrasts: same pair, P!=Q, short
shown=0
for pid,gr in df.groupby("pair_id"):
    if shown>=3: break
    a=gr[gr.label=="high"].iloc[0]; b=gr[gr.label=="low"].iloc[0]
    if a.reply!=b.reply and a["r_words"]<=30:
        print(f"\\n=== {pid} | S={a.S} | H={a.H}\\nP: {a.reply}\\nQ: {b.reply}"); shown+=1"""),
md("Duplicate-check method note: exact-string `duplicated()` on reply/S/H; near-dups not checked (would need embeddings/SimHash)."),
md("## 6. Verdict"),
code("""wm=abs(df[df.label=='high'].r_words.mean()-df[df.label=='low'].r_words.mean())
rep=df.reply; leak=sum(rep.str.lower().str.contains(re.escape(p),regex=True).sum() for p in ["gullible","credulous","skeptic","naive","critical thinking","test point"])
print(f"rows={len(df)} pairs={df.pair_id.nunique()} | P={int((df.label=='high').sum())} Q={int((df.label=='low').sum())}")
print(f"integrity viol={(df.groupby('pair_id').size()!=2).sum()} split-viol=int check above | size viol: S>2={int((df.s_lines>2).sum())} H>2={int((df.h_sents>2).sum())} R>2s={int((df.r_sents>2).sum())}")
print(f"indistinguishability: wordmeanΔ={wm:.3f} {'PASS' if wm<=0.15 else 'FAIL'} | leakage hits={leak}")
print("SHIP" if (wm<=0.15 and leak==0) else "FIX-WHAT: see failed gate above")
print("What-to-add: (1) near-dup audit (2) human spot-check of hedge quality (3) length-cap enforcement if R>2s>0")"""),
]
nbf.write(nb, "notebooks/eda_final.ipynb")
print("wrote notebooks/eda_final.ipynb")
