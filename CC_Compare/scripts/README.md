# 统一运行与核验入口

从工作区根目录运行：

```powershell
python -B CC_Compare/scripts/run_classical.py --help
python -B CC_Compare/scripts/verify_layout.py
```

run_classical.py仅委托现有static_profit入口的classical子命令，执行同目标ALNS/VNS。示例参数为--fold 0 --role validation --case c205 --seconds 10 --iterations 100000 --out CC_Compare/results/new_validation；每次使用新输出目录。它不开放正式外层测试。

其余方法的资格见[方法配置](../configs/methods.json)，尚未完成适配者不伪装为可运行的利润基线。verify_layout.py核对目录、官方源码、数据镜像、权重与科研冻结；加--full-archive可同时重新核验旧目录全部原件SHA。
