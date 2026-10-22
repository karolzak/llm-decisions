import json
s=json.load(open('artifacts/probes/decision-1/summary.json',encoding='utf-8'))
d=json.load(open('docs/results/2026-10-20-decision1-probes/summary-decision-1.json',encoding='utf-8'))
print("same:", s==d)
print(list(s.keys()))
for p,v in s['probes'].items():
    print("==",p, json.dumps(v.get('result'), ensure_ascii=False)[:3000])
for k in s:
    if k!='probes': print("TOP",k, json.dumps(s[k], ensure_ascii=False)[:3000])
