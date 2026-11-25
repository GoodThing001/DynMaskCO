"""Check the ten prospectively selected freight days in the canonical Euclidean cold-chain profile."""
from itertools import combinations
import json
import math
from pathlib import Path

from prepare_panel import core, independent_audit, nearest_neighbour_witness

HERE=Path(__file__).resolve().parent
DAYS=("MO","TU","WE","TH","FR")
RADIUS_KM=6371.0


def arc(a,b):
    lat1,lon1=map(math.radians,a)
    lat2,lon2=map(math.radians,b)
    diff=lon2-lon1
    east=math.cos(lat2)*math.sin(diff)
    north=math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(diff)
    dot=math.sin(lat1)*math.sin(lat2)+math.cos(lat1)*math.cos(lat2)*math.cos(diff)
    angle=math.atan2(math.hypot(east,north),dot)
    return RADIUS_KM*angle,math.atan2(east,north)


def delivery_day(name):
    full=(HERE/"source_external/vrppdl_large"/(name+".txt")).read_text()
    rows=[line.split() for line in full.split("[Nodes/requests]")[1].splitlines()[1:] if line.split()]
    vehicles=full.split("[Vehicle Types]")[1].split("[Number of Vehicle Types]")[0].splitlines()[1:]
    capacity=max(float(line.split()[4]) for line in vehicles if line.split())
    if capacity!=13600:raise ValueError(name)
    hub=next(r for r in rows if r[0]=="0" and r[4]=="-1")
    eligible=[r for r in rows if r[4]=="1" and r[1]=="0" and int(r[8])==0 and int(r[9])==86400]
    if len(eligible)<100:raise ValueError(name)
    records=[hub]+eligible[:100]
    ll=[tuple(map(float,r[2:4])) for r in records]
    xy=[]
    for point in ll:
        rad,bearing=arc(ll[0],point)
        xy.append((rad*math.sin(bearing),rad*math.cos(bearing)))
    nodes=[[i,*xy[i],20000.*int(r[6])/capacity,int(r[10])/60.,int(r[11])/60.,int(r[5])/60.] for i,r in enumerate(records)]
    return nodes,[int(r[0]) for r in records],ll,len(eligible)


def main():
    details={}; sample_coordinates=[]
    for country in ("BE","NL"):
        for day in DAYS:
            name=f"{country}_{day}_0_F"
            nodes,source_ids,ll,eligible=delivery_day(name)
            sample_coordinates.append(set(ll[1:]))
            xy=[r[1:3] for r in nodes]
            error=max(abs(math.dist(xy[i],xy[j])/arc(ll[i],ll[j])[0]-1)
                      for i,j in combinations(range(len(xy)),2) if ll[i]!=ll[j])
            model=core.PublicationModel(dict(nodes=nodes,distance=[[math.dist(a,b) for b in xy] for a in xy],name=name),core.load_config(),"main_Product1",.8)
            bad=[i for i in range(1,101) if not model.route((i,)).feasible]
            num_routes=None
            if not bad:
                init=nearest_neighbour_witness(model,True)
                independent_audit(model,init,True);num_routes=len(init)
            details[name]=dict(eligible=eligible,source_node_ids=source_ids,
                               mean_demand_share=sum(r[3] for r in nodes)/2000000,
                               max_projection_relative_error=error,unserviceable_ids=bad,common_initializer_routes=num_routes)
    max_overlap=max(len(a&b) for a,b in combinations(sample_coordinates,2))
    passed=all(not d["unserviceable_ids"] and d["max_projection_relative_error"]<=.001 for d in details.values())
    result=dict(status="PASS" if passed else "STOP_ADMISSION",solver_endpoints=0,source_cases=details,max_exact_coordinate_overlap=max_overlap)
    (HERE/"freight_admission.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(result["status"],[(c,d["eligible"],len(d["unserviceable_ids"]),round(d["max_projection_relative_error"],7)) for c,d in details.items()],max_overlap)
if __name__=="__main__":main()
