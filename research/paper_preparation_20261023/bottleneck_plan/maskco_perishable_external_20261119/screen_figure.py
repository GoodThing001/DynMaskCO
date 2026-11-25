"""Plot actual source effects and Danish route economics on the development panel."""
from pathlib import Path
from statistics import mean
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent/"results/mixscreen01"

def main():
    scores=json.loads((HERE/"analysis.json").read_text())["sources"]
    rows=json.loads((HERE/"danish_economics.json").read_text())
    with plt.rc_context({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False}):
        fig,(a,b)=plt.subplots(1,2,figsize=(10.6,4.0),layout="constrained",gridspec_kw={"width_ratios":[.83,1.17]})
        for y,(s,label) in enumerate((("GH-p100","GH"),("XTW-p100","XTW"),("PDPCDPG-D100","丹麦"))):
            z=scores[s]["ORDER_SPLIT_D"]
            vals=list(z["by_seed"].values())
            a.plot(vals,[y,y],color="#8ab8ba",lw=2.5)
            a.scatter(vals,[y,y],facecolor="white",edgecolor="#2d7374",s=36,zorder=3)
            a.scatter(z["mean"],y,color="#174c56",marker="D",s=40,zorder=4)
            a.annotate(f"{z['mean']:+.1f}",(z["mean"],y),xytext=(0,12),textcoords="offset points",ha="center",color="#174c56")
        a.set(yticks=[0,1,2],yticklabels=["GH","XTW","丹麦"],ylim=(-.6,2.6),xlim=(-36,180),xlabel="MASKCO 减同规则距离（U）")
        a.set_title("完整 60 秒搜索的净效益",loc="left",fontweight="bold")
        a.scatter([],[],facecolor="white",edgecolor="#2d7374",label="单种子");a.scatter([],[],marker="D",color="#174c56",label="均值")
        a.legend(frameon=False,loc="lower right",ncol=2,handletextpad=.4,columnspacing=.8)
        a.axvline(0,color="#606b70",lw=.8);a.grid(axis="x",alpha=.18)
        x=np.arange(5);keys=sorted(rows)
        for dx,key,name,color in ((-.24,"quality","品质损失","#036765"),(0.,"distance","里程","#c66b36"),(.24,"cold","制冷","#7794aa")):
            v=[mean(r[key] for r in rows[g]) for g in keys]
            b.bar(x+dx,v,width=.22,color=color,label=name,zorder=2)
        b.scatter(x,[mean(r["U"] for r in rows[g]) for g in keys],color="#212a34",marker="D",label="净效益",s=28,zorder=3)
        b.set(xticks=x,xticklabels=[c[-1].upper() for c in keys],ylabel="对净效益的贡献（U）",xlabel="丹麦开发几何",ylim=(-193,213))
        b.set_title("丹麦差额由哪些成本构成",loc="left",fontweight="bold")
        b.axhline(0,color="#606b70",lw=.8);b.grid(axis="y",alpha=.18)
        b.legend(frameon=False,loc="upper left",ncol=4,fontsize=8,columnspacing=.7,handlelength=1.0)
        for ax in (a,b):ax.spines[["right","top"]].set_visible(False)
        fig.savefig(HERE/"配对净效益.png",dpi=210,bbox_inches="tight")
        plt.close(fig)

if __name__=="__main__":main()
