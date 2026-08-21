"""step 3d: confidence-interval coverage"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import OPEN_HF_DATASETS, RESULTS_DIR, paths, topic_outputs_problem
from stats_utils import ci_coverage_check


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=sorted(OPEN_HF_DATASETS), default="de")
    ap.add_argument("--n-rep", type=int, default=1000,
                    help="Monte Carlo repetitions for the primary interval")
    ap.add_argument("--n-bootstrap-rep", type=int, default=120,
                    help="subset of repetitions used for slower bootstrap alternatives")
    ap.add_argument("--n-boot", type=int, default=120)
    args = ap.parse_args()
    p = paths(args.lang)
    if problem := topic_outputs_problem(args.lang):
        sys.exit(problem)

    df = pd.read_csv(p["with_topics_csv"])
    rows = []
    scenario_seed = 42
    for corpus, sub in df.groupby("dataset", sort=True):
        ct = pd.crosstab(sub["cefr_level"], sub["topic"])
        observed = ct.to_numpy(dtype=float)
        observed /= observed.sum()
        independent = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True)
        for scenario, joint in (("observed_joint", observed),
                                ("independence_same_margins", independent)):
            r = ci_coverage_check(
                n=len(sub), joint=joint, n_rep=args.n_rep, n_boot=args.n_boot,
                n_bootstrap_rep=args.n_bootstrap_rep, seed=scenario_seed,
            )
            scenario_seed += 1
            r.update({
                "language": args.lang,
                "corpus": corpus,
                "scenario": scenario,
                "n_levels": int(ct.shape[0]),
                "n_topics": int(ct.shape[1]),
                "n_boot": args.n_boot,
            })
            rows.append(r)
            print(
                f"{corpus:<20} {scenario:<26} n={len(sub):>5} "
                f"true V={r['true_v']:.3f}  ncx2={r['coverage_ncx2']:5.0%}  "
                f"normal={r['coverage_normal']:5.0%}  "
                f"percentile={r['coverage_percentile']:5.0%}"
            )

    cols = [
        "language", "corpus", "scenario", "n", "n_levels", "n_topics",
        "true_v", "n_rep_ncx2", "n_rep_bootstrap", "n_boot",
        "coverage_ncx2", "coverage_normal",
        "coverage_percentile", "coverage_basic",
    ]
    result = pd.DataFrame(rows)[cols]
    out = RESULTS_DIR / f"ci_coverage_{args.lang}.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("Coverage is a simulation diagnostic, not a guarantee. The thesis should")
    print("call the noncentral-chi-square intervals approximate and report the")
    print("permutation tests separately.")


if __name__ == "__main__":
    main()
