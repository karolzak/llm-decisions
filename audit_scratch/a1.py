import json, glob, os
base = r"artifacts/aml/summaries/replication"
for f in sorted(glob.glob(base + "/replication__*.json")):
    j = json.load(open(f, encoding="utf-8"))
    s = j.get("summary") or {}
    print("==", os.path.basename(f), list(s.keys()))
    for k in ("training",):
        if k in s:
            for arm, v in s[k].items():
                print("  training", arm, {kk: (vv if not isinstance(vv, (list, dict)) else type(vv).__name__) for kk, vv in v.items()})
                ho = v.get("holdout")
                if ho:
                    for h in ho: print("    ", h)
    if "reports" in s:
        print("  reports:", list(s["reports"].keys()))
