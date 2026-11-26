"""Print compact markdown tables from the probe JSON outputs in runs/spatial-intelligence-researcher/."""
import json
import sys

from _common import out_dir

D = out_dir()


def dr(fname):
    rep = json.load(open(D / fname))
    print(f"\n### {fname}\n")
    print("| run | ends | est | err@100 | err@250 | err@525 | err@1050 | rel50 | rel200 | cell64 exact | cell64 within1 | cell128 within1 | resid free | resid blocked | blocked share |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, r in rep.items():
        a = r["agg"]
        for e in ("V0", "V1", "V2"):
            def g(k, fmt="{:.0f}"):
                if k not in a:
                    return "-"
                return fmt.format(a[k]["median"]) + (f" (n={a[k]['n']})" if "@" in k else "")
            print(f"| {key} | {r['ends']} | {e} | {g(e+'_err@100')} | {g(e+'_err@250')} | {g(e+'_err@525')} | {g(e+'_err@1050')} | "
                  f"{g(e+'_rel50_mean')} | {g(e+'_rel200_mean')} | {g(e+'_cell64_exact','{:.2f}')} | {g(e+'_cell64_within1','{:.2f}')} | "
                  f"{g(e+'_cell128_within1','{:.2f}')} | {g(e+'_resid_free','{:.2f}')} | {g(e+'_resid_blocked','{:.2f}')} | {g(e+'_resid_share_blocked','{:.2f}')} |")
        print(f"| {key} | | hdg | max_hdg_err median {a['max_hdg_err']['median']:.3f} p90 {a['max_hdg_err']['p90']:.3f} deg; blocked_frac {a['blocked_frac']['median']:.2f}; jumps {a['jumps']['median']:.0f}; p10 V2 err@1050 {a.get('V2_err@1050',{}).get('p10',float('nan')):.0f} p90 {a.get('V2_err@1050',{}).get('p90',float('nan')):.0f} |")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["dead_reckoning.json", "dead_reckoning_nomonsters.json"]:
        if (D / f).exists():
            dr(f)
