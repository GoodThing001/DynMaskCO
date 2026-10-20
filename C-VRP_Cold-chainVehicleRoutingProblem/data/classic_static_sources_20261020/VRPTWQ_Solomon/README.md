# vrptwq-solomon-instances
Benchmark instances for the Vehicle Routing Problem with Time Windows and Quality constraints (VRPTWQ), adapted from [Solomon (1987)](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/).

These instances are used in:

An Exact Algorithm for the Vehicle Routing Problem with Time Windows and Perishable Products

## Instances

The repository contains 56 instances with 100 customers, organized in three classes based on customer geographical distribution: **R** (random), **C** (clustered), and **RC** (mixed). Each class has two time window types: tight (type 1) and wide (type 2).

### File format

Each file contains one line per node (depot + 100 customers), with tab-separated columns:

```
ID    X    Y    DEMAND    TW_START    TW_END    SERVICE_TIME
```

- Node 0 is the depot.
- Coordinates are in km.
- Time windows are in minutes (24-hour planning horizon = 1440 minutes).
- Demands are in kg.
- Service time is in minutes.

### Transformations from original Solomon instances

Following [Dabia et al. (2017)](https://doi.org/10.1287/trsc.2016.0655), the original Solomon instances are scaled to represent B2B distribution operations with long-haul segments and substantial onsite service requirements:

| Parameter | Value |
|-----------|-------|
| Coordinates | Scaled by 3.5 |
| Time windows | Scaled to 24-hour horizon: $a'_i = \frac{24 \cdot 60}{b_0} \cdot a_i$, $b'_i = \frac{24 \cdot 60}{b_0} \cdot b_i$ |
| Demands | Multiplied by 100 |
| Service time | Fixed at 90 minutes (1.5 hours) for all customers |

where $b_0$ is the end of the depot time window in the original instance.

## Additional parameters

The following parameters are not included in the instance files and are defined in the paper:

### Fleet

| Parameter | Value |
|-----------|-------|
| Truck capacity $Q$ | 20,000 kg |
| Average truck speed | 60 km/h |
| Distance cost $c^d$ | 1 €/km |

### Refrigeration and quality decay

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Thermal conductivity | $\lambda$ | 0.36 W/m°C | 
| Surface area | $S$ | 30 m² | 
| Wall thickness | $L$ | 0.1 m |
| External temperature | $T_{ext}$ | 20°C | 
| Regulatory limit | $T_{reg}$ | 7°C | 
| Optimal temperature | $T_{opt}$ | 0°C | 
| Initial quality level | — | 400 |
| Max quality decay | $q_{max}$ | 80 (20% of 400) | 
| Quality decay per stop | $\gamma$ | 1 per stop | 
| Quality decay per time | $\alpha$ | 0.015 per min | 
| Quality decay per temp×time | $\beta$ | 0.0013 per min·°C |
| Quality decay coefficient | $\tau$ | 0.1 per min | 
| Refrigeration cost | $c^e$ | 5×10⁻⁵ €/W·min |
