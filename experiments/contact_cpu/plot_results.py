"""Optional visual check of offline window errors; NOT a closed-loop policy evaluation.

Usage: python plot_results.py results/cpu_trace01_v1
Matplotlib is required only for this script (never for contact_gate/evaluate).
"""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir', type=Path)
    p.add_argument('--max-plots', type=int, default=30)
    args = p.parse_args()
    if args.max_plots < 1:
        p.error('--max-plots must be >= 1')
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless; never requires the remote GPU desktop
        import matplotlib.pyplot as plt
    except ImportError as e:
        p.error('Matplotlib not installed. Optional: pip install matplotlib; no GPU needed.')

    path = args.result_dir / 'frames.csv'
    groups = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            groups[(row['trace'], row['window'])].append(row)
    if not groups:
        raise ValueError(f'No evaluation rows in {path}')
    out = args.result_dir / 'plots'
    out.mkdir(exist_ok=True)
    for i, ((trace, window), rows) in enumerate(groups.items()):
        if i >= args.max_plots:
            break
        rows.sort(key=lambda x: int(x['step']))
        steps = [int(r['step']) for r in rows]
        m0 = [float(r['m0_pos_m']) for r in rows]
        m1 = [float(r['m1_pos_m']) for r in rows]
        m2 = [float(r['m2_selected_pos_m']) if r['m2_selected_pos_m'] else float('nan') for r in rows]
        resolved = sum(r['m2_selected_pos_m'] not in ('', None) for r in rows)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(steps, m0, 'o-', label='M0: hold-last-pose')
        ax.plot(steps, m1, 's-', label='M1: rigid SE(3)')
        if resolved:
            ax.plot(steps, m2, '.-', label='M2: contact gate (resolved steps only)')
        ax.set(xlabel='Hidden step label', ylabel='Position error (coordinate units; m if verified)',
               title=f'Offline position error | window {window} | M2 resolved {resolved}/{len(rows)}')
        ax.legend()
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        ident = hashlib.sha256(trace.encode('utf-8')).hexdigest()[:8]
        target = out / f'trace_{ident}_window_{window.replace(":", "-")}.png'
        fig.savefig(target, dpi=160)
        plt.close(fig)
        print(target)
    if len(groups) > args.max_plots:
        print(f'Skipped {len(groups)-args.max_plots} groups; increase --max-plots if needed.')


if __name__ == '__main__':
    main()
