"""Render the held-out heat contribution, its geometry interval and the full comparison rank."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def read(f):return json.loads(f.read_text(encoding="utf8"))
def draw(rows,path):
    fig,axis=plt.subplots(figsize=(8.3,3.25),layout="constrained")
    for i,(label,d,rank,color) in enumerate(rows):
        y=len(rows)-1-i
        low,high=d["ci95"]
        axis.plot([low,high],[y,y],lw=2,color=color)
        axis.scatter([d["mean"]],[y],s=51,color=color,zorder=3)
        axis.annotate(f"{d['mean']:+.1f} U",(d["mean"],y),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9,color=color)
        axis.text(1.02,y,f"{rank} / 13",transform=axis.get_yaxis_transform(),va="center",weight="bold")
    axis.set(yticks=list(reversed(range(len(rows)))),yticklabels=[r[0] for r in rows],
             ylim=(-.5,len(rows)-.2),xlabel="Net benefit of MASKCO over matched distance (U)")
    axis.text(1.02,len(rows)-.35,"Rank",transform=axis.get_yaxis_transform(),size=9)
    axis.axvline(0,color="#555555",ls="--",lw=1);axis.grid(axis="x",alpha=.2)
    axis.spines[["top","right"]].set_visible(False)
    fig.savefig(path,dpi=215,bbox_inches="tight");plt.close(fig)

def development():
    source=HERE.parent/"maskco_heatmap_calibration_20261119"/"DIAGNOSIS.md"
    comparisons={}
    for line in source.read_text(encoding="utf8").splitlines():
        fields=[x.strip() for x in line.strip("|").split("|")]
        if len(fields)==6 and fields[0] in ("VRPTWQ-C2","XTW-p100","GH-p100") and "%" in fields[4]:
            comparisons[fields[0]]=(float(fields[2].split("（")[0]),float(fields[1]))
    effects=read(HERE/"results/placebodev01/analysis.json")["sources"]
    fig,(timing,alignment)=plt.subplots(1,2,figsize=(9.8,3.25),layout="constrained",
                                        gridspec_kw={"width_ratios":[1.55,1]})
    sources=["VRPTWQ-C2","XTW-p100","GH-p100"]
    for i,s in enumerate(sources):
        y=2-i
        for shift,value,color in ((.15,comparisons[s][0],"#276c62"),(-.15,comparisons[s][1],"#a94838")):
            timing.barh(y+shift,value,height=.24,color=color)
            timing.annotate(f"{value:+.1f}",(value,y+shift),xytext=(3 if value>=0 else -3,0),
                            textcoords="offset points",ha="left" if value>=0 else "right",va="center",size=8)
        val=effects[s]["aligned_shuffled"]["mean"]
        alignment.barh(y,val,height=.37,color="#365f8b")
        alignment.annotate(f"{val:+.1f}",(val,y),xytext=(4,0),textcoords="offset points",va="center",size=9)
    timing.set(yticks=[2,1,0],yticklabels=["VRPTWQ-C2","XTW","GH"],xlim=(-155,338),
               title="Fixed insertion rule",xlabel="MASKCO minus distance (U)")
    alignment.set(yticks=[2,1,0],yticklabels=[],xlim=(0,32),title="New repair operator",
                  xlabel="True minus shuffled heatmap (U)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#276c62",label="Equal repair count"),Patch(color="#a94838",label="Equal search time")],
               loc="outside lower center",ncol=2,frameon=False,fontsize=9)
    for ax in (timing,alignment):
        ax.set_ylim(-.45,2.5);ax.axvline(0,color="#555",lw=.75);ax.grid(axis="x",alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(HERE/"development_effects.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
def main():
    root=HERE/"reserved_validation"
    r=read(root/"results/reserve02/analysis.json");m=read(root/"locked_method.json")["candidate"]
    if r["complete"]!=1053:raise ValueError("Final results incomplete")
    rows=[]
    for source,name in (("PDPCDPG-D100","Perishable deliveries"),("XTW-p100","XTW (registered)"),("GH-p100","GH800")):
        s=r["sources"][source];rows.append((name,s["contrasts"][m][m+"_D"],s["ranks"][m],"#136c63"))
    extra=read(HERE/"xtw_low_overlap/analysis.json")["panels"]["低重叠十例"]
    rows.insert(2,("XTW (low overlap)",extra["contrasts"][m+"_D"],extra["rank"],"#845b36"))
    draw(rows,root/"paired_effects.png")
if __name__=="__main__":
    import sys
    development() if sys.argv[1:]==["development"] else main()
