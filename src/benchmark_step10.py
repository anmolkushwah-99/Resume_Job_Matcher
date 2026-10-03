import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.matching.final_match_engine import FinalMatchEngine, _get_all_job_ids_from_db_or_csv
from src.matching.match_service import get_resume_text, get_job_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_step10_benchmark(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Benchmarks the Step 10 matching engine across multiple operational workloads.
    """
    root_dir = Path(__file__).resolve().parent.parent
    if output_dir is None:
        output_dir = root_dir / "data" / "processed" / "evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = FinalMatchEngine.get_instance()

    test_resume_id = "RES001"
    test_job_id = "JOB001"
    all_job_ids = _get_all_job_ids_from_db_or_csv()
    if not all_job_ids:
        all_job_ids = [f"JOB{i:03d}" for i in range(1, 13)]

    # Warm-up run
    _ = engine.match_one(test_resume_id, test_job_id, persist=False)
    _ = engine.match_many(test_resume_id, job_ids=all_job_ids, persist=False)

    # 1. Benchmark Raw SVM Vectorization + Inference (no DB lookup)
    res_text = get_resume_text(test_resume_id)
    job_text = get_job_text(test_job_id)
    pair_text = f"{res_text} {job_text}"

    raw_latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = engine.classifier.decision_function([pair_text])
        t1 = time.perf_counter()
        raw_latencies.append((t1 - t0) * 1000.0)

    raw_mean_ms = float(np.mean(raw_latencies))
    raw_std_ms = float(np.std(raw_latencies))

    # 2. Benchmark Full Single-Pair Engine (including text resolution & skill explanation)
    single_latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = engine.match_one(test_resume_id, test_job_id, persist=False)
        t1 = time.perf_counter()
        single_latencies.append((t1 - t0) * 1000.0)

    single_mean_ms = float(np.mean(single_latencies))
    single_std_ms = float(np.std(single_latencies))

    # 3. Benchmark Multi-Job Engine (1 resume against 12 benchmark jobs)
    multi_latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = engine.match_many(test_resume_id, job_ids=all_job_ids, persist=False)
        t1 = time.perf_counter()
        multi_latencies.append((t1 - t0) * 1000.0)

    multi_mean_ms = float(np.mean(multi_latencies))
    multi_std_ms = float(np.std(multi_latencies))
    throughput_pairs_per_sec = float((len(all_job_ids) / (multi_mean_ms / 1000.0)))

    # 4. Compile Results DataFrame
    records = [
        {
            "workload": "Raw SVM Inference (Vectorization + Decision Function)",
            "batch_size_pairs": 1,
            "mean_latency_ms": round(raw_mean_ms, 3),
            "std_latency_ms": round(raw_std_ms, 3),
            "throughput_pairs_per_sec": round(1000.0 / max(0.001, raw_mean_ms), 1),
            "description": "Pure in-memory TF-IDF transform and LinearSVC decision_function call"
        },
        {
            "workload": "Full Single-Pair Match Engine (match_one)",
            "batch_size_pairs": 1,
            "mean_latency_ms": round(single_mean_ms, 3),
            "std_latency_ms": round(single_std_ms, 3),
            "throughput_pairs_per_sec": round(1000.0 / max(0.001, single_mean_ms), 1),
            "description": "Document resolution, skill extraction, SVM decision score, and skill gap explanation"
        },
        {
            "workload": "Multi-Job Matching & Ranking Engine (match_many)",
            "batch_size_pairs": len(all_job_ids),
            "mean_latency_ms": round(multi_mean_ms, 3),
            "std_latency_ms": round(multi_std_ms, 3),
            "throughput_pairs_per_sec": round(throughput_pairs_per_sec, 1),
            "description": "Batch TF-IDF vectorization, batch SVM scoring, multi-job ranking, and skill gap breakdown"
        }
    ]

    df = pd.DataFrame(records)
    out_csv = output_dir / "step10_inference_benchmark.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved Step 10 inference benchmark results to: %s", out_csv)

    print("\n" + "=" * 75)
    print("STEP 10 FINAL MATCHING ENGINE INFERENCE BENCHMARK")
    print("=" * 75)
    for r in records:
        print(f"Workload:      {r['workload']}")
        print(f"Batch Size:    {r['batch_size_pairs']} pairs")
        print(f"Latency (avg): {r['mean_latency_ms']:.3f} ms (+/- {r['std_latency_ms']:.3f} ms)")
        print(f"Throughput:    {r['throughput_pairs_per_sec']:.1f} pairs/sec")
        print("-" * 75)

    return df


if __name__ == "__main__":
    run_step10_benchmark()
