"""Paired effects and ranks from the two independent three-source comparisons."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"maskco_heatmap_calibration_20261119"

def plot():
    groups=[(OLD/"results/confirm01",["VRPTWQ-C1","XTW-p100","GH-p100"],"First validation","#53677e"),
            (HERE/"results/replicate01",["PDPCDPG-D100","XTW-p100","GH-p100"],"External replication","#00746a")]
    fig,ax=plt.subplots(figsize=(8.4,4.2),layout="constrained")
    labels=[];ticks=[]
    for j,(folder,sources,label,color) in enumerate(groups):
        data=json.loads((folder/"analysis.json").read_text(encoding="utf8"))
        audit=json.loads((folder/"independent_audit.json").read_text(encoding="utf8"))
        if audit["status"]!="PASS" or audit["endpoints"]!=891:raise ValueError("Incomplete independent audit")
        for i,s in enumerate(sources):
            d=data["sources"][s];h=d["contrasts"]["HYBRID"]["HYBRID_D"]
            y=6-4*j-i
            if h["ci95"] is not None:
                ax.plot(h["ci95"],[y,y],color=color,lw=2)
            ax.scatter(h["mean"],y,marker=("s","o")[j],color=color,s=42,label=label if i==0 else "",zorder=3)
            ax.annotate(f"{h['mean']:+.1f}",(h["mean"],y),xytext=(0,7),textcoords="offset points",
                        ha="center",color=color,fontsize=9)
            names={"VRPTWQ-C1":"VRPTWQ C1","PDPCDPG-D100":"Perishable delivery","XTW-p100":"XTW","GH-p100":"GH400" if j==0 else "GH600"}
            labels.append(f"{names[s]}  ({h['geometries']} {'geometry' if h['geometries']==1 else 'geometries'})")
            ticks.append(y)
            ax.text(1.02,y,f"{d['ranks']['HYBRID']} / 11",transform=ax.get_yaxis_transform(),va="center",ha="left",fontsize=9)
    ax.text(1.02,6.65,"Rank",transform=ax.get_yaxis_transform(),ha="left",weight="bold",fontsize=9)
    ax.set(yticks=ticks,yticklabels=labels,ylim=(-.85,6.95),xlabel="Net benefit from learned heatmap over matched distance (U)")
    ax.axvline(0,color="#555",linewidth=.9,linestyle="--")
    ax.grid(axis="x",alpha=.2)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,-.2))
    fig.savefig(HERE/"independent_effects.png",dpi=220,bbox_inches="tight")
    plt.close(fig)
if __name__=="__main__":plot()
