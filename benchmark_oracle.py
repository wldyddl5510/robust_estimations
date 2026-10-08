"""Serial, reproducible end-to-end oracle ablations with independent certificates.

Examples (run with the repository's Gurobi-enabled Python):
  python benchmark_oracle.py --module-dir /path/to/original-snapshot --variants baseline --output /tmp/baseline.json
  python benchmark_oracle.py --module-dir . --variants exact_indicator hybrid_indicator hybrid_bigm --output /tmp/improved.json

The baseline label calls the supplied source's default settings, so use it only
with an original pre-change source snapshot. Current sources should use an
explicit exact_* or hybrid_* variant. An optional --baseline-dir chooses the
original independent certification implementation; otherwise the tested source's
standalone exact indicator oracle certifies the result independently of the
estimator's heuristic/constraint-generation path.

Each variant gets a fresh interpreter, but import/license warm-up and independent
certification are excluded from the reported end-to-end estimator wall time.
All estimator and certificate solver models use one thread. Data, block seeds,
and projection seeds are identical across variants. Each dataset is run three
times by default; summarize each seed's median before comparing datasets.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


CASES = {
    "sparse_d4_k5": dict(method="sparse", n=120, d=4, s=1, delta=0.1),
    "sparse_d6_k7": dict(method="sparse", n=120, d=6, s=2, delta=0.05),
    "projected_d6_k7": dict(method="projected", n=120, d=6, s=2, delta=0.05, r=2),
    "sparse_d8_k9": dict(method="sparse", n=120, d=8, s=2, delta=0.02),
}
DEFAULT_CASES = list(CASES)[:3]
VARIANTS = {
    "baseline": {},
    "exact_indicator": dict(oracle_strategy="exact", oracle_formulation="indicator", perspective=False),
    "exact_bigm": dict(oracle_strategy="exact", oracle_formulation="bigm", perspective=False),
    "hybrid_bigm": dict(oracle_strategy="hybrid", oracle_formulation="bigm", perspective=False),
    "hybrid_indicator": dict(oracle_strategy="hybrid", oracle_formulation="indicator", perspective=False),
    "exact_perspective": dict(oracle_strategy="exact", oracle_formulation="bigm", perspective=True),
    "hybrid_perspective": dict(oracle_strategy="hybrid", oracle_formulation="bigm", perspective=True),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="Original oracle source for certification; defaults to --module-dir")
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=["exact_indicator", "hybrid_indicator", "hybrid_bigm"])
    p.add_argument("--cases", nargs="+", choices=CASES, default=DEFAULT_CASES)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def run_worker(args):
    import numpy as np
    import gurobipy as gp

    source_files = ("IP_algorithm.py", "projected_algorithm.py", "utils.py")
    source_hashes = {f: hashlib.sha256((args.module_dir / f).read_bytes()).hexdigest() for f in source_files}
    sys.path.insert(0, str(args.module_dir.resolve()))
    import IP_algorithm as ip
    import projected_algorithm as projected
    import utils

    certificate_dir = args.baseline_dir or args.module_dir
    spec = importlib.util.spec_from_file_location("independent_certificate_oracle", certificate_dir / "IP_algorithm.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    certificate_signature = inspect.signature(baseline.separation_oracle).parameters
    certificate_options = {}
    if "formulation" in certificate_signature:
        certificate_options["formulation"] = "indicator"
    if "perspective" in certificate_signature:
        certificate_options["perspective"] = False
    variant = args.variants[0]
    if variant == "baseline" and "oracle_strategy" in inspect.signature(ip.ip_estimation).parameters:
        raise ValueError("The baseline label requires an original pre-change --module-dir; "
                         "use an explicit exact_* or hybrid_* variant for updated sources.")
    variant_kwargs = VARIANTS[variant]
    records = []
    # Warm imports, environment startup, and a trivial optimization before timing.
    with contextlib.redirect_stdout(io.StringIO()):
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)
            env.start()
            with gp.Model(env=env) as model:
                model.Params.OutputFlag = 0
                model.addVar()
                model.optimize()

    for case_name in args.cases:
        case = CASES[case_name]
        for seed in args.seeds:
            rng = np.random.default_rng(17000 + seed)
            true_mean = np.zeros(case["d"])
            true_mean[:case["s"]] = 1.5 * np.where(np.arange(case["s"]) % 2, -1, 1)
            # Multivariate t_5 with covariance I (one chi-square per observation).
            data = true_mean + rng.standard_normal((case["n"], case["d"])) * np.sqrt(
                3 / rng.chisquare(5, size=(case["n"], 1)))
            # Fixed count within the declared 2% replacement contamination budget.
            outlier_rows = rng.choice(case["n"], 2, replace=False)
            data[outlier_rows] = 6 + rng.standard_normal((2, case["d"]))
            kwargs = dict(s=case["s"], epsilon=0.02, lambda_upper=1, delta=case["delta"],
                          tol=args.tol, seed=27000 + seed, C=2, **variant_kwargs)
            for repetition in range(args.repeats):
                dense_runs = []
                original_dense = projected.dense_estimation

                def capture_dense(block_means, center, s, tol, **kw):
                    estimate, info = original_dense(block_means, center, s, tol, **kw)
                    dense_runs.append((block_means, estimate, s, tol, info))
                    return estimate, info

                projected.dense_estimation = capture_dense
                try:
                    started = perf_counter()
                    if case["method"] == "sparse":
                        estimate, info = ip.ip_estimation(data, **kwargs)
                    else:
                        estimate, info = projected.projected_estimation(data, r=case["r"], **kwargs)
                    elapsed = perf_counter() - started
                finally:
                    projected.dense_estimation = original_dense

                certificate_started = perf_counter()
                with gp.Env(empty=True) as env:
                    env.setParam("OutputFlag", 0)
                    env.setParam("Threads", 1)
                    env.start()
                    if case["method"] == "sparse":
                        block_means, *_ = utils.mom_initialization(data, case["s"], 0.02, 1,
                            case["delta"], args.tol, 27000 + seed, C=2)
                        dense_runs = [(block_means, estimate, case["s"], args.tol, info)]
                    certificates = []
                    for block_means, candidate, s, tol, local_info in dense_runs:
                        lower, upper, _ = baseline.separation_oracle(
                            block_means, candidate, s, min(1e-6, tol / 100), env=env,
                            **certificate_options)
                        tolerance = args.tol if case["method"] == "sparse" else args.tol / 4
                        certificates.append({
                            "lower": lower, "upper": upper,
                            "reported_upper": local_info["objective"],
                            "reported_lower": local_info["lower_bound"],
                            "certified_optimality_gap": max(0, upper - local_info["lower_bound"]),
                            "upper_valid": bool(lower <= local_info["objective"] + 2e-6),
                            "gap_valid": bool(upper - local_info["lower_bound"] <= tolerance + 2e-6),
                        })
                record = dict(variant=variant, case=case_name, seed=seed, repetition=repetition,
                    wall_seconds=elapsed, estimate=estimate.tolist(), true_mean=true_mean.tolist(),
                    l2_error=float(np.linalg.norm(estimate - true_mean)), info=info,
                    certificates=certificates, certificate_seconds=perf_counter() - certificate_started)
                records.append(record)
                print(f"{variant} {case_name} seed={seed} repeat={repetition}: {elapsed:.4f}s "
                      f"calls={info.get('oracle_calls')} gap={info.get('gap'):.3g}", file=sys.stderr, flush=True)
    return {
        "configuration": {"variant": variant, "cases": {c: CASES[c] for c in args.cases},
            "seeds": args.seeds, "repeats": args.repeats, "tol": args.tol,
            "epsilon": 0.02, "C": 2, "distribution": "multivariate t5, covariance I, 2/120 replacements",
            "module_dir": str(args.module_dir.resolve()), "certificate_module_dir": str(certificate_dir.resolve()),
            "certificate_implementation": {"function": "separation_oracle", "options": certificate_options,
                "mode": "standalone exact oracle with no decision threshold"},
            "source_sha256": source_hashes,
            "source_changed_during_run": any(hashlib.sha256((args.module_dir / f).read_bytes()).hexdigest()
                != source_hashes[f] for f in source_files),
            "certificate_source_sha256": {filename: hashlib.sha256((certificate_dir / filename).read_bytes()).hexdigest()
                for filename in ("IP_algorithm.py", "projected_algorithm.py", "utils.py")},
            "gurobi_version": list(gp.gurobi.version()), "python": sys.version, "threads": 1},
        "records": records,
    }


def main():
    args = parser().parse_args()
    if args.worker:
        result = run_worker(args)
    else:
        results = []
        for variant in args.variants:
            child_output = args.output.with_name(args.output.stem + "." + variant + ".json")
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                "--module-dir", str(args.module_dir.resolve()),
                "--variants", variant, "--cases", *args.cases, "--seeds", *map(str, args.seeds),
                "--repeats", str(args.repeats), "--tol", str(args.tol), "--output", str(child_output)]
            if args.baseline_dir is not None:
                command.extend(["--baseline-dir", str(args.baseline_dir.resolve())])
            subprocess.run(command, check=True)
            results.append(json.loads(child_output.read_text()))
        result = {"runs": results}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
