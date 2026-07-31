"""E2/E3 analysis: loss curves, optimizer-step timing, trajectory distance
from the gold (eigh-every-step) run."""
import json, torch, os

R = '/home/navin/shampoo/research/rfshampoo'
runs = {}
for tag in ['eigh1', 'eighK', 'rf', 'adamw']:
    try:
        runs[tag] = json.load(open(f'{R}/run_{tag}.json'))
    except FileNotFoundError:
        print(f"[missing run_{tag}.json]")

print(f"{'run':7s} {'median opt ms':>13} {'total wall s':>12} {'final loss':>11} "
      f"{'mean loss last50':>17}")
for tag, r in runs.items():
    med = sorted(r['opt_ms'])[len(r['opt_ms']) // 2]
    last50 = sum(r['loss'][-50:]) / 50
    print(f"{tag:7s} {med:13.0f} {r['wall_s']:12.0f} {r['loss'][-1]:11.4f} "
          f"{last50:17.4f}")

if os.path.exists(f'{R}/snaps_eigh1.pt'):
    gold = torch.load(f'{R}/snaps_eigh1.pt')
    for tag in ['rf', 'eighK', 'adamw']:
        fp = f'{R}/snaps_{tag}.pt'
        if not os.path.exists(fp):
            continue
        s = torch.load(fp)
        ks = sorted(set(gold) & set(s))
        errs = [float((s[k] - gold[k]).norm() / gold[k].norm()) for k in ks]
        print(f"traj |{tag} - gold|/|gold| at steps {ks}: "
              + " ".join(f"{e:.2e}" for e in errs))
