# 统一运行与核验入口

从工作区根目录运行：

```powershell
python -B CC_Compare/scripts/run_classical.py --help
python -B CC_Compare/scripts/run_validation_baselines.py --help
python -B CC_Compare/scripts/run_constructive.py --help
python -B CC_Compare/scripts/verify_layout.py
```

run_classical.py仅委托现有static_profit入口的classical子命令，执行同目标ALNS/VNS。示例参数为--fold 0 --role validation --case c205 --seconds 10 --iterations 100000 --out CC_Compare/results/new_validation；每次使用新输出目录。它不开放正式外层测试。

run_validation_baselines.py委托新增固定第0折四例验证面板，--group main运行ALNS/VNS，--group controls运行距离控制，可用--methods指定该组内的方法。要求进程恰好绑定1个CPU，BLAS/OpenMP线程为1；每次--out使用新目录。服务器执行与取回工具位于扩展tools/baseline_comparison_remote.py；本轮run01、run02、run03已经存在，禁止覆盖重跑。所有结果继续经过严格评价器与独立审计，不开放test角色。

run_constructive.py委托扩展comparison_methods/run_construct.py，支持六种真实作者网络的train/validation；明确指定--device、--cpu、--fold 0与--seed 1000。validation必须提供checkpoint与预算receipt。不得覆盖已存在输出，正式测试不开放。实际命令、环境锁、检查点和开发限制见[完整执行报告](../results/ten_methods_20261021/run01/README.md)。

各方法资格见[方法配置](../configs/methods.json)。verify_layout.py核对目录、官方源码、数据镜像、权重与科研冻结；加--full-archive可同时重新核验旧目录全部原件SHA。
