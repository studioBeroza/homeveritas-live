import sys, pickle, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'bt')
from bee_bt import *
from multiprocessing import Pool
ms=pickle.load(open('bt/ms.pkl','rb'))
V={"A":{}, "B":{"usePart":False}, "C":{"usePD":True}, "D":{"useHtfFvg":True}, "E":{"entryLvl":"ce"}, "F":{"biasMode":"both"}}
def sub_stats(trs, eq, a, z):
    t=[x for x in trs if a<=x['et'].year<=z]
    e=eq[(eq.index.year>=a)&(eq.index.year<=z)]; e=e-e.iloc[0]
    s=stats(t,e); return {k:s.get(k) for k in ("n","win","pf","avgR","maxDDpct","net","exits")}
def job(arg):
    tf,k=arg
    tr,eq=run(ms,tf=tf,**V[k])
    out={"IS":sub_stats(tr,eq,2010,2015),"OOS":sub_stats(tr,eq,2016,2020),"ALL":sub_stats(tr,eq,2010,2020)}
    yr={y:round(sum(x['pnl']/x['risk_usd'] for x in tr if x['et'].year==y),2) for y in range(2010,2021)}
    pickle.dump((tr,eq),open(f'bt/res_{tf}_{k}.pkl','wb'))
    r={"tf":tf,"v":k,**out,"yearR":yr}
    open(f'bt/res_{tf}_{k}.json','w').write(json.dumps(r,default=float))
    return r
if __name__=="__main__":
    jobs=[(tf,k) for tf in ("5min","1min") for k in V]
    with Pool(min(len(jobs),8)) as p:
        for r in p.imap_unordered(job,jobs): print(r["tf"],r["v"],"done",flush=True)
