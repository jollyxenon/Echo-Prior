# Echo-QAOA: Constructive Spin-Echo Response Design and Per-Qubit Marginal Distillation for Variational Quantum Optimization

Official open-source implementation of **Echo-QAOA** and **Echo-Prior** (*Echo-QAOA: Constructive Spin-Echo Response Design and Per-Qubit Marginal Distillation for Variational Quantum Optimization*).

This repository provides the complete codebase for:
1. **Intra-Circuit Spin-Echo Operator Parity Decoupling & Constructive 1D Response Recipes**:
   - Exact cancellation of quadratic spin-spin couplings $H_J$ within the unmodified standard $p$-layer global QAOA circuit ansatz at positive even depth $p=2k$ via complete spin-echo pairs $(\gamma_{2j-1},\gamma_{2j})=(-g_j,g_j)$ and $(\beta_{2j-1},\beta_{2j})=(\pi/2,\theta_j)$, factorizing target-state probabilities into $O(kn)$ single-qubit magnetization responses.
   - Three constructive 1D spectral response recipes: **`Flat-Top`** (Daubechies power-complementary spectral factorization & layerwise lattice reduction with $O(\delta^{2k})$ peak error tolerance and strict convexity on zero-free intervals), **`Alternating-Phase`**, and **`Endpoint-Matched`**.
2. **Zero-Measurement 1D Distribution-Overlap Decoding & Per-Qubit Marginal Distillation (`Echo-Prior`)**:
   - Diagnosis of the conditional implicit-function curvature–sensitivity trade-off and zero-probability cross-branch discontinuities in direct $2p$-angle regression.
   - Permutation-equivariant graph message-passing network (**`Echo-Prior`**: `Echo-Prior-Flat`, `Echo-Prior-End`, `Echo-Prior-Alt`) that distills single-qubit positive-spin marginals $\pi_i^{\mathrm T}\in[0,1]$ of multi-body quantum teacher states from raw Ising problem coefficients $(J,h)$, paired with a 128-point analytical 1D distribution-overlap decoder (**0 Stage-1 quantum circuit evaluations**) and Stage-2 full-angle L-BFGS-B energy recoupling.
   - Zero-shot bidirectional cross-scale generalization across $n\in\{6,8,10,12,14\}$ (5,120 unseen MWIS test instances) and $100\%$ immunity to Stage-1 projective measurement shot noise.

---

## Installation

Dependencies are managed via [`pixi`](https://pixi.sh/) (`default` environment for instance generation, teacher collection, and offline probability benchmarks; `learning` environment for PyTorch training, inference, and energy evaluation; `figures` environment for manuscript figure generation):

```bash
pixi install
pixi install -e learning
pixi install -e figures
```

---

## Quick Start

### 1. Constructive 1D Echo Parameter Generation & Known-Target Comparison

```bash
# Generate analytical 1D Flat-Top Echo parameters and verify against exact statevector simulation
pixi run generate-echo --instance data/benchmark/mwis/mwis_n08_00.npz --depth 8 --variant Flat-Top

# Compare 6 offline probability optimization methods (3 Echo recipes + RI / TQA / INTERP)
pixi run compare-echo --instance data/benchmark/mwis/mwis_n08_00.npz --depth 8 --cap 30 --output runs/example
```

### 2. Online Solving from Raw $(J,h)$ via Frozen `Echo-Prior` Weights

```bash
# Predict per-qubit marginals, run 128-point 1D overlap decoding (0 Stage-1 quantum calls), and refine all 2p angles via L-BFGS-B
pixi run -e learning predict-learning --instance data/benchmark/mwis/mwis_n08_00.npz --checkpoint runs/models/prior_flat_p08/model.pt
```

### 3. Offline Teacher Collection, `Echo-Prior` Training & Evaluation

```bash
# Collect and pack 6-method multi-branch probability teacher labels
pixi run label-learning collect --dataset data/mwis/n10/dataset.npz --depth 8 --cap 10 --workers 16 --output runs/labels/n10-p08
pixi run label-learning pack --dataset data/mwis/n10/dataset.npz --depth 8 --cap 10 --labels runs/labels/n10-p08

# Train Per-Qubit Quantum Teacher Marginal Distillation Network (Echo-Prior-Flat / Echo-Prior-End / Echo-Prior-Alt)
pixi run -e learning train-prior --dataset data/mwis/n10/dataset.npz --labels runs/labels/n10-p08/labels.npz --method Flat-Top --seed 0 --output runs/models/prior_flat_p08

# Train 2p-angle regression baselines (MLP / GNN, seed=0)
pixi run -e learning train-learning --dataset data/mwis/n10/dataset.npz --labels runs/labels/n10-p08/labels.npz --architecture GNN --method TQA-LBFGSB --seed 0 --output runs/models/n10-p08/GNN__TQA-LBFGSB__seed0

# Evaluate frozen models on held-out test instances
pixi run -e learning evaluate-learning --dataset data/mwis/n10/dataset.npz --models runs/models/prior_flat_p08 --split test --output runs/evaluation/prior_flat_p08

# Evaluate online baselines (TQA, RI (1), RI (32), Echo-Energy, Warm-Start QAOA, ma-QAOA)
pixi run -e learning evaluate-baselines --dataset data/mwis/n10/dataset.npz --depth 8 --method TQA --split test --output runs/baselines/tqa_p08

# Run canonical statistical summaries and generate manuscript figures
pixi run -e learning canonical-experiments --task all
pixi run -e figures paper1-figures
```

---

## Manuscript & Repository Structure

- [`paper/paper1_preprint.pdf`](paper/paper1_preprint.pdf) / [`paper/paper1_preprint.tex`](paper/paper1_preprint.tex) / [`paper/paper1-icml-en.md`](paper/paper1-icml-en.md) / [`paper/paper1-icml-zh.md`](paper/paper1-icml-zh.md): Compiled ICML two-column LaTeX preprint (`paper1_preprint.pdf` / `paper1_preprint.tex`), full bilingual Markdown manuscripts, figures (`paper/figures/`), and verified BibTeX (`paper/references.bib`).
- `src/echo/`: Constructive 1D Echo parameter synthesis (`core.py`, `flat_synthesis.py`, `flat_scan.py`, `variants.py`).
- `src/compare/`: Exact statevector evolution, reverse-mode autodiff, and offline probability comparison (`autodiff.py`, `methods.py`, `optimizers.py`, `runtime.py`, `verify.py`).
- `src/learning/`: `Echo-Prior` equivariant GNN, 1D distribution-overlap decoder, direct $2p$-angle regression models, and Stage-2 energy optimization (`echo_prior.py`, `echo_decoder.py`, `echo_prior_pipeline.py`, `angle_models.py`, `optimization.py`, `evaluation.py`).
- `src/instances/`: Five-family benchmark and MWIS dataset generators (`benchmark.py`, `mwis.py`).

---

## Citation

```bibtex
@article{yao2026echoqaoa,
  title   = {{Echo-QAOA}: Constructive spin-echo response design and per-qubit marginal distillation for variational quantum optimization},
  author  = {Yao, Jianxuan},
  journal = {Research Square preprint},
  year    = {2026}
}
```
