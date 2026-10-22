import json, collections
d=json.load(open('artifacts/reports/replication/data-test.json',encoding='utf-8'))
m=d['metrics']
for k,v in m.items():
    if 'decision' in k.lower() or 'b1' in k.lower() or 'msft' in k.lower():
        print(k, {kk:v[kk] for kk in ('accuracy','ece','T','nll') if kk in v})
print([k for k in m])
rows=[json.loads(l) for l in open('data/processed/ace/test.jsonl',encoding='utf-8')]
print(rows[0].keys())
for key in ('label','gold','answer','target'):
    if key in rows[0]:
        c=collections.Counter(r[key] for r in rows); print(key, c, {k:round(v/len(rows),3) for k,v in c.items()})
