import json, sys
rows = json.load(open(sys.argv[1]))
hdr = ("| scheme | lr | k | noise | step/ulp | σ/ulp | rel L2 err vs ‖z‖ | bias (ulp) ± se | std (ulp) | pred SR/RTN | eval-cycle drift std (ulp) / frac changed |")
print(hdr)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ec = r.get("evalcycle_drift_ulp_std")
    ecs = f"{ec:.3f} / {r['evalcycle_frac_changed']:.4f}" if ec is not None else "—"
    print(f"| {r['scheme']} | {r['lr']:.0e} | {r['k']:g} | {r['kind'][:5]} | {r['step_over_ulp']:.3f} | {r['sigma_over_ulp']:.2f} | "
          f"{r['rel_l2_vs_z']:.4f} | {r['bias_ulp']:+.3f} ± {r['bias_se_ulp']:.3f} | {r['std_ulp']:.2f} | "
          f"{r['pred_std_sr']:.1f}/{r['pred_std_rtn']:.1f} | {ecs} |")
