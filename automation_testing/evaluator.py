import json, math, numpy as np
from pathlib import Path
from typing import List, Tuple, Set
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bertscore

GOLD = Path(__file__).with_name("gold.jsonl")
PREDS = Path(__file__).with_name("preds.jsonl")
REPORT = Path(__file__).with_name("report.json")

def read_jsonl(p):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def base_id(x:str)->str:
    return (x or "").split("#",1)[0]

def dcg(rels: List[int]) -> float:
    return sum(rel / math.log2(i+2) for i,rel in enumerate(rels))

def ndcg_at_k(pred: List[str], gold: Set[str], k:int)->float:
    top=pred[:k]
    rels=[1 if base_id(x) in gold else 0 for x in top]
    idcg=dcg(sorted(rels, reverse=True))
    return 0.0 if idcg==0 else dcg(rels)/idcg

def recall_at_k(pred: List[str], gold: Set[str], k:int)->float:
    if not gold: return 0.0
    return len({base_id(x) for x in pred[:k]} & gold)/len(gold)

class Eval:
    def __init__(self, thresh=0.70):
        self.model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.thresh=thresh
    def sbert(self, ans:str, ref:str)->float:
        em=self.model.encode([ans,ref], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(em[0], em[1]).cpu().numpy())
    def nugget_prf(self, nuggets: List[str], ans: str)->Tuple[float,float,float]:
        if not nuggets: return (1.0,1.0,1.0)
        a=self.model.encode([ans], convert_to_tensor=True, normalize_embeddings=True)[0]
        ns=self.model.encode(nuggets, convert_to_tensor=True, normalize_embeddings=True)
        sims=util.cos_sim(ns, a).cpu().numpy().flatten()
        tp=int((sims>=self.thresh).sum()); fn=len(nuggets)-tp; fp=0
        p=tp/(tp+fp) if (tp+fp) else 1.0
        r=tp/(tp+fn) if (tp+fn) else 1.0
        f=0.0 if p+r==0 else 2*p*r/(p+r)
        return p,r,f
    def bert_f1(self, ref:str, ans:str)->float:
        _,_,F1=bertscore([ans],[ref],lang="en",rescale_with_baseline=True,verbose=False)
        return float(F1[0].item())

def main():
    gold={r["id"]:r for r in read_jsonl(GOLD)}
    preds={r["id"]:r for r in read_jsonl(PREDS)}
    ev=Eval()

    per, agg = [], {k:[] for k in ["nugP","nugR","nugF1","sbert","bsF1","R@1","R@3","R@5","N@1","N@3","N@5"]}
    for qid,g in gold.items():
        p=preds.get(qid, {"model_answer":"","retrieved_ids":[]})
        ans=p["model_answer"]
        nugP,nugR,nugF1 = ev.nugget_prf(g.get("nuggets",[]), ans)
        sbert = ev.sbert(ans, g.get("reference_answer",""))
        bsF1  = ev.bert_f1(g.get("reference_answer",""), ans)
        gold_set=set(g.get("gold_passages",[]))
        r1,r3,r5=(recall_at_k(p["retrieved_ids"], gold_set, k) for k in (1,3,5))
        n1,n3,n5=(ndcg_at_k(p["retrieved_ids"], gold_set, k) for k in (1,3,5))
        row={"id":qid,"nugget_precision":nugP,"nugget_recall":nugR,"nugget_f1":nugF1,
             "sbert_cosine":sbert,"bertscore_f1":bsF1,
             "recall@1":r1,"recall@3":r3,"recall@5":r5,
             "ndcg@1":n1,"ndcg@3":n3,"ndcg@5":n5}
        per.append(row)
        for k,v in zip(["nugP","nugR","nugF1","sbert","bsF1","R@1","R@3","R@5","N@1","N@3","N@5"],
                       [nugP,nugR,nugF1,sbert,bsF1,r1,r3,r5,n1,n3,n5]):
            agg[k].append(v)

    avg=lambda xs: float(np.mean(xs)) if xs else 0.0
    summary={
        "count": len(per),
        "nugget_precision": avg(agg["nugP"]),
        "nugget_recall":    avg(agg["nugR"]),
        "nugget_f1":        avg(agg["nugF1"]),
        "sbert_cosine":     avg(agg["sbert"]),
        "bertscore_f1":     avg(agg["bsF1"]),
        "recall@1": avg(agg["R@1"]), "recall@3": avg(agg["R@3"]), "recall@5": avg(agg["R@5"]),
        "ndcg@1":  avg(agg["N@1"]),  "ndcg@3":  avg(agg["N@3"]),  "ndcg@5":  avg(agg["N@5"]),
    }
    REPORT.write_text(json.dumps({"per_question": per, "summary": summary}, indent=2), encoding="utf-8")
    print("Wrote", REPORT)

if __name__=="__main__":
    main()
