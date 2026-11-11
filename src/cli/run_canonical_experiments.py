"""Echo-Prior (Paper 1) 标准实验执行与统计汇总脚本。

涵盖标准子任务：
  1. n10_recompute    : 基于 runs/models_and_eval/ 评测记录，以 Bootstrap(4000, seed=0) 重算 n=10 神经模型、基线与 1D 流形消融汇总；
  2. cross_scale_mwis : 在 data/mwis/ 跨规模测试集 n∈{6,8,10,12,14}（每规模 1,024 题，p=10）上评测 Echo-Prior-Flat (seed=0) 双向零样本跨规模泛化与在线对照基线；
  3. shot_noise       : 在 data/mwis/n10/dataset.npz 测试集 1,024 题（p=10）上评测 Echo-Prior-Flat 与 TQA 的有限采样散粒噪声模式 A 与模式 B。
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basis import flips_for, spin_table
from compare.autodiff import evolve
from instances.benchmark import energy_table
from learning.aggregation import aggregate, bootstrap_ci
from learning.baselines import METHODS as BASELINE_METHODS, run_ma_qaoa, run_warm_start_qaoa, select_initial
from learning.echo_decoder import decode, expand_scalar
from learning.echo_prior import build_features, echo_prior_probabilities, train_and_predict_manifold_ablations
from learning.echo_prior_pipeline import restore as restore_prior
from learning.optimization import EnergyProblem, fast_dist, shot_noise_adam_optimize
from learning.scoring import score_angles
from records import append_jsonl, cache_path, read_jsonl, save_json, write_jsonl

PRIOR_FLAT_SEED0_WEIGHTS = {
    depth: ROOT / f"runs/models_and_eval/models/n10-p{depth:02d}__Echo-Prior__Flat-Top__seed0/model.pt"
    for depth in (8, 10, 12)
}

PRIOR_FLAT_SEED0_EVALS = {
    depth: ROOT / f"runs/models_and_eval/evaluation/n10-p{depth:02d}__Echo-Prior__Flat-Top__seed0.jsonl"
    for depth in (8, 10, 12)
}

SHOT_LEVELS_STAGE1 = [16, 32, 64, 128, 256, 512, 1024]
SHOT_LEVELS_PARAMETER_SHIFT = [64, 256, 1024, 4096]


def summarize_pct(values: np.ndarray, replicates: int = 4000, seed: int = 0) -> dict:
    """对百分比数组执行 Bootstrap(4000, seed=0)，同时返回绝对 95% 区间与相对均值的上下偏移量。"""
    arr = np.asarray(values, dtype=np.float64)
    mean_val, lo, hi = bootstrap_ci(arr, replicates=replicates, seed=seed)
    return dict(
        mean_pct=round(mean_val, 2),
        ci95=[round(lo, 2), round(hi, 2)],
        ci_offset=[round(mean_val - lo, 2), round(hi - mean_val, 2)],
    )


def summarize_paired_diff(target_pct: np.ndarray, base_pct: np.ndarray,
                          replicates: int = 4000, seed: int = 0) -> dict:
    """对两组同题百分比得分差 (target - base) 执行 Bootstrap(4000, seed=0)。"""
    diff = np.asarray(target_pct, dtype=np.float64) - np.asarray(base_pct, dtype=np.float64)
    mean_diff, lo, hi = bootstrap_ci(diff, replicates=replicates, seed=seed)
    return dict(
        mean_diff_pp=round(mean_diff, 2),
        ci95=[round(lo, 2), round(hi, 2)],
        ci_offset=[round(mean_diff - lo, 2), round(hi - mean_diff, 2)],
        win_rate_pct=round(float(np.mean(diff > 0.0) * 100.0), 1),
    )


def load_jsonl_sorted(path: Path) -> list[dict]:
    """读取 JSONL 文件并按题号 id 升序排列。"""
    return sorted(read_jsonl(path), key=lambda row: str(row["id"]))


def load_mwis_benchmark_instances(n: int, count: int) -> list[dict]:
    """加载跨规模测试题库（n∈{6,8,10,12,14} 来自 data/mwis/n{n:02d}/dataset.npz 的 test 划分）。"""
    mwis_npz = ROOT / f"data/mwis/n{n:02d}/dataset.npz"
    if mwis_npz.exists():
        with np.load(mwis_npz, allow_pickle=False) as raw:
            test_idx = np.flatnonzero(raw["split"] == "test")[:count]
            if len(test_idx) != count:
                raise ValueError(f"{mwis_npz} 中 test 划分题数 {len(test_idx)} 不足请求的 {count}")
            return [
                dict(
                    index=int(idx),
                    id=str(raw["ids"][idx]),
                    J=raw["J"][idx].astype(np.float64),
                    h=raw["h"][idx].astype(np.float64),
                    target_index=int(raw["target_index"][idx]),
                    s_star=raw["s_star"][idx].astype(np.float64),
                )
                for idx in test_idx
            ]
    raise FileNotFoundError(f"未找到跨规模题库 {mwis_npz}")


def predict_echo_prior_flat_seed0(depth: int, J_batch: np.ndarray, h_batch: np.ndarray) -> np.ndarray:
    """使用 Echo-Prior-Flat (seed=0) 标准冻结权重对任意规模 (J_batch, h_batch) 进行单次零样本前向推理。"""
    model, _ = restore_prior(PRIOR_FLAT_SEED0_WEIGHTS[depth])
    features = build_features(
        np.asarray(J_batch, dtype=np.float64),
        np.asarray(h_batch, dtype=np.float64),
    ).astype(np.float32)
    return echo_prior_probabilities(model, features)


def stage2_block(initial_angles, optimized: dict, score: dict,
                 stage1_calls: int, stage1_seconds: float, **extra) -> dict:
    """canonical 逐方法记录的唯一字段口径。"""
    return dict(
        initial_angles=initial_angles, angles=optimized["angles"],
        initial_probability=score["initial_probability"], probability=score["probability"],
        initial_energy=score["initial_energy"], energy=score["energy"],
        stage1_calls=int(stage1_calls), stage2_calls=int(optimized["calls"]),
        stage1_seconds=float(stage1_seconds), stage2_seconds=float(optimized["seconds"]),
        total_seconds=float(stage1_seconds) + float(optimized["seconds"]),
        status=optimized["status"], **extra,
    )


def _eval_online_baseline_method(
    method: str,
    inst_idx: int,
    J: np.ndarray,
    h: np.ndarray,
    target_idx: int,
    depth: int,
    prob: EnergyProblem,
    beta: float = 2.0,
) -> dict:
    """在单道实例上按在线基线方法执行无监督求解与评分。"""
    if method in ("Warm-Start QAOA", "ma-QAOA"):
        res = (run_warm_start_qaoa(J, h, depth, target_idx, beta_mf=beta)
               if method == "Warm-Start QAOA" else run_ma_qaoa(J, h, depth, target_idx, seed=0))
        return dict(initial_angles=res["initial_angles"], angles=res["angles"],
                    initial_probability=res["initial_probability"], probability=res["probability"],
                    initial_energy=res["initial_energy"], energy=res["energy"],
                    stage1_calls=res["stage1_calls"], stage2_calls=res["stage2_calls"],
                    stage1_seconds=res["stage1_seconds"], stage2_seconds=res["stage2_seconds"],
                    total_seconds=res["stage1_seconds"] + res["stage2_seconds"],
                    status=res["status"])
    method_idx = BASELINE_METHODS.index(method)
    rng = np.random.default_rng([0, int(inst_idx), depth, method_idx])
    stage1_started = time.perf_counter()
    init_ang, calls, extra = select_initial(method, depth, J, h, prob, rng, grid_points=128)
    stage1_seconds = time.perf_counter() - stage1_started
    optimized = prob.optimize(init_ang)
    score = score_angles(prob.energy, prob.flips, target_idx, init_ang, optimized["angles"])
    return stage2_block(init_ang.tolist(), optimized, score, calls, stage1_seconds, **extra)


def _eval_single_baseline_task(task: tuple) -> dict:
    """在单道测试题上评测单一基线方法。"""
    method, inst_idx, inst_id, J, h, target_idx, depth = task
    prob = EnergyProblem(J, h)
    res = _eval_online_baseline_method(method, inst_idx, J, h, target_idx, depth, prob)
    return dict(id=str(inst_id), index=int(inst_idx), **res)


def _train_ablation_depth_task(task: tuple) -> tuple[int, np.ndarray, np.ndarray | None]:
    """在独立进程中训练单个深度的一维流形表示消融模型并返回测试集预测标量。"""
    p, X_train, h_train, J_train, X_test, run_grid = task
    pi_train = predict_echo_prior_flat_seed0(p, J_train, h_train)
    pred_scalar_a, pred_grid_a = train_and_predict_manifold_ablations(
        X_train, h_train, pi_train, X_test, n_qubits=J_train.shape[1],
        depth=p, seed=0, run_grid=run_grid,
    )
    return p, pred_scalar_a, pred_grid_a


def _eval_ablation_scalar_task(task: tuple) -> dict:
    """在单道测试题上从给定 1D 平顶标量展开并运行 Stage-2 能量优化。"""
    inst_idx, inst_id, J, h, target_idx, depth, scalar_a = task
    prob = EnergyProblem(J, h)
    init_ang = expand_scalar("Flat-Top", depth, float(scalar_a))
    optimized = prob.optimize(init_ang)
    score = score_angles(prob.energy, prob.flips, target_idx, init_ang, optimized["angles"])
    return dict(
        id=str(inst_id),
        index=int(inst_idx),
        **stage2_block(init_ang.tolist(), optimized, score, 0, 0.0, scalar=float(scalar_a)),
    )


def ensure_baseline_records(eval_root: Path, workers: int = 24) -> None:
    """确保 n=10 测试集上的无先验基线与 1D 流形消融（p=8, 10, 12）评测记录完整生成。"""
    baselines_dir = eval_root / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    baseline_specs = {
        "Echo-Energy": "Echo-Energy__p{p:02d}.jsonl",
        "TQA": "TQA__p{p:02d}.jsonl",
        "RI (1)": "RI-1__p{p:02d}.jsonl",
        "RI (32)": "RI-32__p{p:02d}.jsonl",
    }
    raw_mwis = np.load(ROOT / "data/mwis/n10/dataset.npz", allow_pickle=False)
    train_idx = np.flatnonzero(raw_mwis["split"] == "train")
    test_idx = np.flatnonzero(raw_mwis["split"] == "test")
    J_train = raw_mwis["J"][train_idx].astype(np.float64)
    h_train = raw_mwis["h"][train_idx].astype(np.float64)
    J_test = raw_mwis["J"][test_idx].astype(np.float64)
    h_test = raw_mwis["h"][test_idx].astype(np.float64)
    target_test = raw_mwis["target_index"][test_idx].astype(int)
    ids_test = raw_mwis["ids"][test_idx].astype(str)

    X_train = build_features(J_train, h_train).astype(np.float32)
    X_test = build_features(J_test, h_test).astype(np.float32)

    ablation_jobs = []
    for p in (8, 10, 12):
        for method, pat in baseline_specs.items():
            out_file = baselines_dir / pat.format(p=p)
            if out_file.exists():
                records = read_jsonl(out_file)
                if len(records) == len(test_idx):
                    continue
            print(f"  [Baseline] Generating {method} (p={p}) on n=10 test instances ({len(test_idx)} items)...", flush=True)
            tasks = [
                (method, int(test_idx[i]), ids_test[i], J_test[i], h_test[i], int(target_test[i]), p)
                for i in range(len(test_idx))
            ]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(_eval_single_baseline_task, tasks, chunksize=16))
            records.sort(key=lambda r: int(r["index"]))
            write_jsonl(out_file, records)

        f_scalar = baselines_dir / f"Echo-GNN-Scalar__p{p:02d}.jsonl"
        f_grid = baselines_dir / f"Echo-GNN-Grid__p{p:02d}.jsonl"
        has_scalar = f_scalar.exists() and len(read_jsonl(f_scalar)) == len(test_idx)
        has_grid = f_grid.exists() and len(read_jsonl(f_grid)) == len(test_idx)
        if not (has_scalar and has_grid):
            ablation_jobs.append((p, X_train, h_train, J_train, X_test, not has_grid))

    if ablation_jobs:
        with ProcessPoolExecutor(max_workers=min(len(ablation_jobs), workers)) as pool:
            trained_list = list(pool.map(_train_ablation_depth_task, ablation_jobs))
        for p, pred_scalar_a, pred_grid_a in trained_list:
            f_scalar = baselines_dir / f"Echo-GNN-Scalar__p{p:02d}.jsonl"
            f_grid = baselines_dir / f"Echo-GNN-Grid__p{p:02d}.jsonl"
            for out_file, a_preds in ((f_scalar, pred_scalar_a), (f_grid, pred_grid_a)):
                if a_preds is None or (out_file.exists() and len(read_jsonl(out_file)) == len(test_idx)):
                    continue
                tasks = [
                    (int(test_idx[i]), ids_test[i], J_test[i], h_test[i], int(target_test[i]), p, float(a_preds[i]))
                    for i in range(len(test_idx))
                ]
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    records = list(pool.map(_eval_ablation_scalar_task, tasks, chunksize=16))
                records.sort(key=lambda r: int(r["index"]))
                write_jsonl(out_file, records)


def run_task_n10_recompute(out_dir: Path, workers: int = 24) -> dict:
    """以 Bootstrap(4000, seed=0) 重算 n=10 神经模型、基线与 3 组 1D 流形消融汇总。"""
    print("=== [Task: n10_recompute] Recomputing n=10 summaries with Bootstrap(4000, seed=0) ===", flush=True)
    eval_root = ROOT / "runs/models_and_eval"
    groups = aggregate(eval_root, replicates=4000, seed=0)

    ensure_baseline_records(eval_root, workers=workers)
    baselines_dir = eval_root / "baselines"
    baseline_specs = {
        "Echo-Energy": "Echo-Energy__p{p:02d}.jsonl",
        "TQA": "TQA__p{p:02d}.jsonl",
        "RI (1)": "RI-1__p{p:02d}.jsonl",
        "RI (32)": "RI-32__p{p:02d}.jsonl",
    }
    manifold_specs = {
        "Echo-GNN-Scalar": "Echo-GNN-Scalar__p{p:02d}.jsonl",
        "Echo-GNN-Grid": "Echo-GNN-Grid__p{p:02d}.jsonl",
    }

    raw_mwis = np.load(ROOT / "data/mwis/n10/dataset.npz", allow_pickle=False)
    test_idx = np.flatnonzero(raw_mwis["split"] == "test")
    J_test = raw_mwis["J"][test_idx].astype(np.float64)
    h_test = raw_mwis["h"][test_idx].astype(np.float64)
    s_star_test = raw_mwis["s_star"][test_idx].astype(np.float64)

    baselines_summary = {}
    manifold_summary = {}
    product_state_decomposition = {}

    for p in (8, 10, 12):
        p_key = f"p{p:02d}"
        baselines_summary[p_key] = {}
        manifold_summary[p_key] = {}

        prior_rows = load_jsonl_sorted(PRIOR_FLAT_SEED0_EVALS[p])
        prior_init = np.array([r["initial_probability"] * 100.0 for r in prior_rows])
        prior_fin = np.array([r["probability"] * 100.0 for r in prior_rows])
        prior_calls = float(np.mean([r["stage2_calls"] for r in prior_rows]))

        pred_pi = predict_echo_prior_flat_seed0(p, J_test, h_test)
        ideal_prod_prob = np.prod(np.where(s_star_test > 0, pred_pi, 1.0 - pred_pi), axis=1) * 100.0
        product_state_decomposition[p_key] = dict(
            ideal_prior_product=summarize_pct(ideal_prod_prob),
            echo_1d_initial=summarize_pct(prior_init),
            stage2_refined=summarize_pct(prior_fin),
            stage2_vs_initial=summarize_paired_diff(prior_fin, prior_init),
        )

        tqa_rows = load_jsonl_sorted(baselines_dir / baseline_specs["TQA"].format(p=p))
        tqa_fin = np.array([r["probability"] * 100.0 for r in tqa_rows])
        for name, pattern in baseline_specs.items():
            rows = load_jsonl_sorted(baselines_dir / pattern.format(p=p))
            init_arr = np.array([r["initial_probability"] * 100.0 for r in rows])
            fin_arr = np.array([r["probability"] * 100.0 for r in rows])
            calls_mean = float(np.mean([r["stage2_calls"] for r in rows]))
            baselines_summary[p_key][name] = dict(
                initial=summarize_pct(init_arr),
                final=summarize_pct(fin_arr),
                mean_calls=round(calls_mean, 1),
                vs_tqa=summarize_paired_diff(fin_arr, tqa_fin),
            )

        manifold_summary[p_key]["Echo-Prior-Flat"] = dict(
            initial=summarize_pct(prior_init),
            final=summarize_pct(prior_fin),
            mean_calls=round(prior_calls, 1),
            vs_prior_flat=summarize_paired_diff(prior_fin, prior_fin),
        )
        for name, pattern in manifold_specs.items():
            spec_file = baselines_dir / pattern.format(p=p)
            if not spec_file.exists():
                manifold_summary[p_key][name] = None
                continue
            rows = load_jsonl_sorted(spec_file)
            init_arr = np.array([r["initial_probability"] * 100.0 for r in rows])
            fin_arr = np.array([r["probability"] * 100.0 for r in rows])
            calls_mean = float(np.mean([r["stage2_calls"] for r in rows]))
            manifold_summary[p_key][name] = dict(
                initial=summarize_pct(init_arr),
                final=summarize_pct(fin_arr),
                mean_calls=round(calls_mean, 1),
                vs_prior_flat=summarize_paired_diff(fin_arr, prior_fin),
                prior_flat_gain_vs_this=summarize_paired_diff(prior_fin, fin_arr),
            )

    report = dict(
        bootstrap_replicates=4000,
        seed=0,
        neural_groups=groups,
        baselines=baselines_summary,
        manifold_1d_ablation=manifold_summary,
        product_state_decomposition=product_state_decomposition,
    )
    save_json(out_dir / "n10_recompute_summary.json", report)
    return report


def _eval_online_scaling_worker(task: tuple) -> dict:
    """在单道实例上评测 Echo-Prior-Flat (seed=0) 与在线对照基线。"""
    inst, depth, pi_prior = task
    J, h, target_idx, s_star = inst["J"], inst["h"], inst["target_index"], inst["s_star"]
    n = len(h)
    prob = EnergyProblem(J, h)

    out = dict(index=int(inst["index"]), id=str(inst["id"]), n=n)
    stage1_started = time.perf_counter()
    scalar, ang, overlap = decode("Flat-Top", depth, h, pi_prior, grid_points=128)
    stage1_seconds = time.perf_counter() - stage1_started
    optimized = prob.optimize(ang)
    score = score_angles(prob.energy, prob.flips, target_idx, ang, optimized["angles"])
    s_pred = np.where(pi_prior >= 0.5, 1.0, -1.0)
    out["Echo-Prior-Flat"] = stage2_block(
        ang.tolist(), optimized, score, 0, stage1_seconds,
        scalar=scalar, distribution_overlap=overlap,
        bit_acc_pct=float(np.mean(s_pred == s_star) * 100.0),
    )
    for b_method in BASELINE_METHODS:
        out[b_method] = _eval_online_baseline_method(
            b_method, int(inst["index"]), J, h, target_idx, depth, prob
        )
    return out


def run_task_cross_scale_mwis(out_dir: Path, workers: int = 24, depth: int = 10,
                              sizes: tuple[tuple[int, int], ...] = (
                                  (6, 1024), (8, 1024), (10, 1024), (12, 1024), (14, 1024),
                              )) -> dict:
    """运行在线 MWIS 跨规模实验（n∈{6,8,10,12,14} 各 1,024 题, p=10，Prior-Flat seed=0，Bootstrap 4000 seed=0）。"""
    print(f"=== [Task: cross_scale_mwis] Running Online MWIS Cross-Scale Generalization (p={depth}) ===", flush=True)
    summary = {"bootstrap_replicates": 4000, "seed": 0, "scales": {}}

    for n, count in sizes:
        instances = load_mwis_benchmark_instances(n, count)
        t1 = time.perf_counter()
        on_cache = cache_path(out_dir, f"cache_prior_cross_scale_mwis_n{n:02d}_p{depth:02d}")
        on_recs = read_jsonl(on_cache) if on_cache.exists() else []
        if len(on_recs) != count:
            done_dict = {int(r["index"]): r for r in on_recs}
            J_stack = np.stack([inst["J"] for inst in instances])
            h_stack = np.stack([inst["h"] for inst in instances])
            prior_list = predict_echo_prior_flat_seed0(depth, J_stack, h_stack)
            remaining_tasks = [
                (instances[i], depth, prior_list[i])
                for i in range(count)
                if int(instances[i]["index"]) not in done_dict
            ]
            if remaining_tasks:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_eval_online_scaling_worker, t) for t in remaining_tasks]
                    for fut in as_completed(futures):
                        r = fut.result()
                        done_dict[int(r["index"])] = r
                        append_jsonl(on_cache, r)
            on_recs = [done_dict[int(inst["index"])] for inst in instances]
            on_recs.sort(key=lambda r: int(r["index"]))
            write_jsonl(on_cache, on_recs)

        tqa_f_on = np.array([r["TQA"]["probability"] for r in on_recs]) * 100.0
        scale_entry = {"n": n, "count": count, "seconds": round(time.perf_counter() - t1, 1), "methods": {}}
        for m_name in ("Echo-Prior-Flat",) + BASELINE_METHODS:
            mi = np.array([r[m_name]["initial_probability"] for r in on_recs]) * 100.0
            mf = np.array([r[m_name]["probability"] for r in on_recs]) * 100.0
            mcalls = float(np.mean([r[m_name]["stage2_calls"] for r in on_recs]))
            si = summarize_pct(mi)
            sf = summarize_pct(mf)
            sd = summarize_paired_diff(mf, tqa_f_on)
            scale_entry["methods"][m_name] = dict(
                init_pct=si["mean_pct"], init_ci95=si["ci95"],
                final_pct=sf["mean_pct"], final_ci95=sf["ci95"],
                vs_tqa_diff_pp=sd["mean_diff_pp"], vs_tqa_ci95=sd["ci95"],
                mean_calls=round(mcalls, 1),
            )
        summary["scales"][f"n{n:02d}"] = scale_entry

    save_json(out_dir / "cross_scale_mwis_p10.json", summary)
    return summary


def run_stage1_shot_noise_instance(task: tuple) -> dict:
    """模式 A：单道测试题上 TQA（32 候选点）在有限采样 N_shots 下的选点压力测试。"""
    idx, inst_id, J, h, target_idx, depth = task
    prob = EnergyProblem(J, h)
    energy = prob.energy
    flips = prob.flips
    u = (np.arange(depth) + 0.5) / depth

    ladder_1d = np.geomspace(0.05, 5.0, 32)
    cands_1d = [np.r_[dt * u, (-dt * (1.0 - u) + np.pi / 2.0) % np.pi - np.pi / 2.0] for dt in ladder_1d]
    dists_1d = np.array([np.abs(evolve(ang, energy, flips)) ** 2 for ang in cands_1d])
    exact_e_1d = dists_1d @ energy

    rng = np.random.default_rng(np.random.SeedSequence([0, 1, int(idx)]))
    n_trials = 5
    selected_1d = {"inf": [int(np.argmin(exact_e_1d))]}

    for shots in SHOT_LEVELS_STAGE1:
        s1_list = []
        for _ in range(n_trials):
            noisy_e_1d = [float((rng.multinomial(shots, p_dist) @ energy) / shots) for p_dist in dists_1d]
            s1_list.append(int(np.argmin(noisy_e_1d)))
        selected_1d[str(shots)] = s1_list

    unique_1d = set(m for m_list in selected_1d.values() for m in m_list)
    cache_1d = {}
    for m in unique_1d:
        init_p = float(dists_1d[m, target_idx])
        opt = prob.optimize(cands_1d[m])
        fin_st = evolve(np.asarray(opt["angles"], dtype=np.float64), energy, flips)
        fin_p = float(np.abs(fin_st[target_idx]) ** 2)
        cache_1d[m] = (init_p, fin_p, float(opt["energy"]), int(opt["calls"]))

    out_1d = {}
    exact_best_1d = selected_1d["inf"][0]
    for key, m_list in selected_1d.items():
        vals = [cache_1d[m] for m in m_list]
        out_1d[key] = dict(
            initial_probability=float(np.mean([v[0] for v in vals])),
            probability=float(np.mean([v[1] for v in vals])),
            energy=float(np.mean([v[2] for v in vals])),
            stage2_calls=float(np.mean([v[3] for v in vals])),
            exact_candidate_match=float(np.mean([m == exact_best_1d for m in m_list])),
        )
    return dict(index=int(idx), id=str(inst_id), n=len(h), TQA=out_1d)


def run_parameter_shift_instance(task: tuple) -> dict:
    """模式 B：单道测试题上调用 shot_noise_adam_optimize（真实两点移位电路投影测量）评测 TQA 与 Echo-Prior-Flat。"""
    idx, inst_id, J, h, target_idx, depth, prior_flat_ang = task
    energy = energy_table(spin_table(len(h)), J, h, 0.0)
    flips = flips_for(len(h))
    u = (np.arange(depth) + 0.5) / depth

    ladder_1d = np.geomspace(0.05, 5.0, 32)
    cands_1d = [np.r_[dt * u, (-dt * (1.0 - u) + np.pi / 2.0) % np.pi - np.pi / 2.0] for dt in ladder_1d]
    dists_1d = np.array([fast_dist(ang, energy, flips) for ang in cands_1d])

    opt_steps = 64
    res = {}
    for shots in list(SHOT_LEVELS_PARAMETER_SHIFT) + [None]:
        if shots is None:
            rng = None
            exact_e_1d = dists_1d @ energy
            tqa_init = cands_1d[int(np.argmin(exact_e_1d))]
            s_key = "inf"
        else:
            rng = np.random.default_rng(np.random.SeedSequence([0, 2, int(shots), int(idx)]))
            noisy_e_1d = [float((rng.multinomial(shots, p) @ energy) / shots) for p in dists_1d]
            tqa_init = cands_1d[int(np.argmin(noisy_e_1d))]
            s_key = str(shots)

        res[s_key] = {}
        for m_name, init_ang in [
            ("TQA", tqa_init),
            ("Echo-Prior-Flat", prior_flat_ang),
        ]:
            init_p = float(fast_dist(init_ang, energy, flips)[target_idx])
            snaps = shot_noise_adam_optimize(
                init_ang, energy, flips, shots=shots, steps=opt_steps, rng=rng
            )
            p16 = float(fast_dist(snaps[16], energy, flips)[target_idx])
            p32 = float(fast_dist(snaps[32], energy, flips)[target_idx])
            p64 = float(fast_dist(snaps[64], energy, flips)[target_idx])
            res[s_key][m_name] = dict(initial_probability=init_p, step16_probability=p16,
                                      step32_probability=p32, probability=p64)

    return dict(index=int(idx), id=str(inst_id), n=len(h), results=res)


def run_task_shot_noise(out_dir: Path, workers: int = 24, depth: int = 10) -> dict:
    """运行有限采样散粒噪声实验（在 data/mwis/n10/dataset.npz 1,024 题上同步评测 Echo-Prior-Flat 与 TQA，Bootstrap 4000 seed=0）。"""
    print(f"=== [Task: shot_noise] Running Shot-Noise Stress Tests on 1,024 test instances (p={depth}) ===", flush=True)
    raw = np.load(ROOT / "data/mwis/n10/dataset.npz", allow_pickle=False)
    test_idx = np.flatnonzero(raw["split"] == "test")
    J_all = raw["J"].astype(np.float64)
    h_all = raw["h"].astype(np.float64)
    target_all = raw["target_index"].astype(int)
    ids_all = raw["ids"].astype(str)

    def _rec_idx(r: dict) -> int:
        return int(r["index"]) if "index" in r else int(r["id"].split("_")[-1])

    prior_jsonl = PRIOR_FLAT_SEED0_EVALS[depth]
    prior_recs = load_jsonl_sorted(prior_jsonl)
    prior_init_arr = np.array([r["initial_probability"] * 100.0 for r in prior_recs])
    prior_final_arr = np.array([r["probability"] * 100.0 for r in prior_recs])
    prior_init_angles = {_rec_idx(r): np.array(r["initial_angles"], dtype=np.float64) for r in prior_recs}

    stage1_cache = cache_path(out_dir, f"stage1_shot_noise_p{depth:02d}")
    s1_records = read_jsonl(stage1_cache) if stage1_cache.exists() else []
    if len(s1_records) != len(test_idx):
        tasks_s1 = [
            (int(idx), str(ids_all[idx]), J_all[idx], h_all[idx], int(target_all[idx]), depth)
            for idx in test_idx
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            s1_records = list(pool.map(run_stage1_shot_noise_instance, tasks_s1, chunksize=16))
        s1_records.sort(key=lambda r: int(r["index"]))
        write_jsonl(stage1_cache, s1_records)

    ps_cache = cache_path(out_dir, f"end_to_end_parameter_shift_p{depth:02d}")
    ps_records = read_jsonl(ps_cache) if ps_cache.exists() else []
    if len(ps_records) != len(test_idx):
        tasks_ps = [
            (
                int(idx),
                str(ids_all[idx]),
                J_all[idx],
                h_all[idx],
                int(target_all[idx]),
                depth,
                prior_init_angles[int(idx)],
            )
            for idx in test_idx
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            ps_records = list(pool.map(run_parameter_shift_instance, tasks_ps, chunksize=16))
        ps_records.sort(key=lambda r: int(r["index"]))
        write_jsonl(ps_cache, ps_records)

    pr_i_s = summarize_pct(prior_init_arr)
    pr_f_s = summarize_pct(prior_final_arr)

    summary = dict(
        depth=depth,
        test_instances=len(test_idx),
        bootstrap_replicates=4000,
        seed=0,
        zero_shot_selection_references={
            "Echo-Prior-Flat": dict(
                stage1_quantum_evals=0,
                seed=0,
                mean_init_pct=pr_i_s["mean_pct"],
                init_ci95=pr_i_s["ci95"],
                mean_final_pct=pr_f_s["mean_pct"],
                final_ci95=pr_f_s["ci95"],
                final_ci_minus=pr_f_s["ci_offset"][0],
                final_ci_plus=pr_f_s["ci_offset"][1],
            ),
        },
        regime_a_stage1_shot_noise={"TQA": {}},
        regime_b_end_to_end_parameter_shift={},
    )

    s1_records.sort(key=lambda r: str(r["id"]))
    for shots_key in [str(s) for s in SHOT_LEVELS_STAGE1] + ["inf"]:
        init_v = np.array([r["TQA"][shots_key]["initial_probability"] * 100.0 for r in s1_records])
        fin_v = np.array([r["TQA"][shots_key]["probability"] * 100.0 for r in s1_records])
        match_v = np.array([r["TQA"][shots_key]["exact_candidate_match"] * 100.0 for r in s1_records])
        si = summarize_pct(init_v)
        sf = summarize_pct(fin_v)
        pr_diff = summarize_paired_diff(prior_final_arr, fin_v)
        summary["regime_a_stage1_shot_noise"]["TQA"][shots_key] = dict(
            mean_init_pct=si["mean_pct"],
            init_ci95=si["ci95"],
            init_ci=si["ci_offset"],
            mean_final_pct=sf["mean_pct"],
            final_ci95=sf["ci95"],
            final_ci=sf["ci_offset"],
            exact_candidate_match_pct=round(float(np.mean(match_v)), 2),
            Echo_Prior_Flat_net_gain_pp=pr_diff["mean_diff_pp"],
            Echo_Prior_Flat_gain_ci95=pr_diff["ci95"],
        )

    ps_records.sort(key=lambda r: str(r["id"]))
    evals_per_step = 4 * depth + 1
    for shots in list(SHOT_LEVELS_PARAMETER_SHIFT) + [None]:
        s_key = "inf" if shots is None else str(shots)
        summary["regime_b_end_to_end_parameter_shift"][s_key] = {}
        for m_name, s1_evals in [("TQA", 32), ("Echo-Prior-Flat", 0)]:
            init_v = np.array([r["results"][s_key][m_name]["initial_probability"] * 100.0 for r in ps_records])
            p16_v = np.array([r["results"][s_key][m_name]["step16_probability"] * 100.0 for r in ps_records])
            p32_v = np.array([r["results"][s_key][m_name]["step32_probability"] * 100.0 for r in ps_records])
            fin_v = np.array([r["results"][s_key][m_name]["probability"] * 100.0 for r in ps_records])
            si = summarize_pct(init_v)
            s16 = summarize_pct(p16_v)
            s32 = summarize_pct(p32_v)
            sf = summarize_pct(fin_v)
            s2_evals_16 = 1 + 16 * evals_per_step
            s2_evals_32 = 1 + 32 * evals_per_step
            s2_evals_64 = 1 + 64 * evals_per_step
            summary["regime_b_end_to_end_parameter_shift"][s_key][m_name] = dict(
                stage1_evals=s1_evals,
                stage2_initial_evals=1,
                stage2_evals_per_step=evals_per_step,
                stage2_evals_16steps=s2_evals_16,
                stage2_evals_32steps=s2_evals_32,
                stage2_evals_64steps=s2_evals_64,
                total_quantum_shots_16steps=None if shots is None else (s1_evals + s2_evals_16) * shots,
                total_quantum_shots_32steps=None if shots is None else (s1_evals + s2_evals_32) * shots,
                total_quantum_shots_64steps=None if shots is None else (s1_evals + s2_evals_64) * shots,
                mean_init_pct=si["mean_pct"],
                init_ci95=si["ci95"],
                init_ci=si["ci_offset"],
                step16_pct=s16["mean_pct"],
                step16_ci95=s16["ci95"],
                step16_ci=s16["ci_offset"],
                step32_pct=s32["mean_pct"],
                step32_ci95=s32["ci95"],
                step32_ci=s32["ci_offset"],
                mean_final_pct=sf["mean_pct"],
                final_ci95=sf["ci95"],
                final_ci=sf["ci_offset"],
            )

        tqa_64 = np.array([r["results"][s_key]["TQA"]["probability"] * 100.0 for r in ps_records])
        t64 = np.array([r["results"][s_key]["Echo-Prior-Flat"]["probability"] * 100.0 for r in ps_records])
        t32 = np.array([r["results"][s_key]["Echo-Prior-Flat"]["step32_probability"] * 100.0 for r in ps_records])
        t16 = np.array([r["results"][s_key]["Echo-Prior-Flat"]["step16_probability"] * 100.0 for r in ps_records])
        d64 = summarize_paired_diff(t64, tqa_64)
        d32 = summarize_paired_diff(t32, tqa_64)
        d16 = summarize_paired_diff(t16, tqa_64)
        summary["regime_b_end_to_end_parameter_shift"][s_key]["Echo-Prior-Flat_vs_TQA_pp"] = dict(
            mean_diff_pp=d64["mean_diff_pp"],
            ci95=d64["ci95"],
            step32_vs_64step_base_pp=d32["mean_diff_pp"],
            step32_ci95=d32["ci95"],
            step16_vs_64step_base_pp=d16["mean_diff_pp"],
            step16_ci95=d16["ci95"],
        )

    save_json(out_dir / f"shot_noise_summary_p{depth:02d}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "n10_recompute", "cross_scale_mwis", "shot_noise"),
        default="all",
        help="选择要执行的标准评测子任务",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/canonical")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "n10_recompute"):
        run_task_n10_recompute(out_dir, workers=args.workers)
    if args.task in ("all", "cross_scale_mwis"):
        run_task_cross_scale_mwis(out_dir, workers=args.workers, depth=args.depth)
    if args.task in ("all", "shot_noise"):
        run_task_shot_noise(out_dir, workers=args.workers, depth=args.depth)


if __name__ == "__main__":
    main()
