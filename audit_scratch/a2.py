import json, glob, os
base = r"artifacts/aml/summaries/replication"
for f in sorted(glob.glob(base + "/*.json")):
    j = json.load(open(f, encoding="utf-8"))
    s = j.get("summary") or {}
    runs = s.get("runs") or {}
    for ds, rr in runs.items():
        for rid, splits in rr.items():
            for sp, m in splits.items():
                if not isinstance(m, dict): continue
                print(f"RUN {os.path.basename(f)[:40]:40s} {ds:9s} {rid:45s} {sp:11s} n={m.get('n')} acc={m.get('accuracy')} ece={m.get('ece')} T={m.get('T')} ece_cal={m.get('ece_cal')}")
    comp = s.get("compare") or {}
    for ds, rr in comp.items():
        for rid, splits in rr.items():
            for sp, c in splits.items():
                if not isinstance(c, dict): continue
                print(f"CMP {os.path.basename(f)[:40]:40s} {ds:9s} {rid:45s} {sp:11s} kappa={c.get('kappa')} mixref={c.get('prediction_mix_reference')} mixrun={c.get('prediction_mix_run')}")
