from pathlib import Path
p = Path("scripts/p5v2_figures.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})''',
'''plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def _setup_fonts():
    """优先使用系统 CJK 字体，避免中文标题缺字。"""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "Arial Unicode MS"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return cand
    return None


CJK_FONT = _setup_fonts()''')
p.write_text(t, encoding="utf-8")
print("font setup added")
