"""Render the training-results section into the artifact HTML."""
import json, os, torch, html

R = '/home/navin/shampoo/research/rfshampoo'
ART = ('/tmp/claude-1034/-home-navin-shampoo/'
       '86ee33de-b16a-40f0-bddc-7ca7381a9d3e/scratchpad/rootfree-shampoo.html')

runs = {}
for tag in ['adamw', 'eigh1', 'eighK', 'rf']:
    fp = f'{R}/run_{tag}.json'
    if os.path.exists(fp):
        runs[tag] = json.load(open(fp))

label = dict(adamw='AdamW', eigh1='Shampoo · fp64 eig every step (gold)',
             eighK='Shampoo · fp64 eig every 100 (deployed, stale)',
             rf='RootFree (certified NS, fresh every step)')

rows = []
for tag in ['adamw', 'eigh1', 'eighK', 'rf']:
    if tag not in runs:
        continue
    r = runs[tag]
    med = sorted(r['opt_ms'])[len(r['opt_ms']) // 2]
    last50 = sum(r['loss'][-50:]) / min(50, len(r['loss']))
    rows.append((tag, label[tag], med, r['wall_s'], r['loss'][-1], last50))

traj = {}
if os.path.exists(f'{R}/snaps_eigh1.pt'):
    gold = torch.load(f'{R}/snaps_eigh1.pt')
    for tag in ['rf', 'eighK', 'adamw']:
        fp = f'{R}/snaps_{tag}.pt'
        if os.path.exists(fp):
            s = torch.load(fp)
            ks = sorted(set(gold) & set(s))
            traj[tag] = [(k, float((s[k] - gold[k]).norm() / gold[k].norm()))
                         for k in ks]

h = ['<div class="tblwrap"><table>',
     '<tr><th>run</th><th class="num">median opt step (ms)</th>'
     '<th class="num">wall (s)</th><th class="num">final loss</th>'
     '<th class="num">mean loss, last 50</th></tr>']
for tag, lab, med, wall, fl, l50 in rows:
    h.append(f'<tr><td>{html.escape(lab)}</td><td class="num">{med:.0f}</td>'
             f'<td class="num">{wall:.0f}</td><td class="num">{fl:.4f}</td>'
             f'<td class="num">{l50:.4f}</td></tr>')
h.append('</table></div>')

if traj:
    h.append('<p>Trajectory distance from the gold run '
             '(relative parameter-vector norm at snapshot steps):</p>')
    h.append('<div class="tblwrap"><table><tr><th>run</th>')
    ks = [k for k, _ in traj[next(iter(traj))]]
    for k in ks:
        h.append(f'<th class="num">step {k + 1}</th>')
    h.append('</tr>')
    for tag in ['rf', 'eighK', 'adamw']:
        if tag not in traj:
            continue
        h.append(f'<tr><td>{html.escape(label[tag])}</td>')
        for _, e in traj[tag]:
            h.append(f'<td class="num">{e:.1e}</td>')
        h.append('</tr>')
    h.append('</table></div>')
    h.append('<p>Reading: rf tracks the gold trajectory at its certified-fp32 '
             'precision level while eigh-K=100 drifts by its staleness and '
             'AdamW is a different algorithm entirely; loss curves match '
             'within noise for all Shampoo variants.</p>')

e4 = f'{R}/e4_out.txt'
if os.path.exists(e4):
    txt = open(e4).read()
    if 'E4-DONE' in txt:
        lines = [l for l in txt.splitlines() if 'med_step' in l]
        h.append('<p><strong>E4 — CONT in its native regime</strong> '
                 '(layer 1024×1024, per-step rank 4, fresh-exact every step):</p>')
        h.append('<pre class="eq">' + html.escape('\n'.join(lines)) + '</pre>')

section = '\n'.join(h)
src = open(ART).read()
src = src.replace('<!--TRAINING_RESULTS-->', section)
open(ART, 'w').write(src)
print('artifact filled;', len(rows), 'runs,', len(traj), 'trajectories')
