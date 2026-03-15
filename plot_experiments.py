import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 1 – Plotting Notebook

    Loads results from `./results/` (produced by `train_experiments.py`) and
    generates all figures.

    - **Exp 1**: AULC bar chart + learning curves per target env
    - **Exp 2**: AULC comparison (frozen-aug vs scratch) for both regimes
    - **Exp 3**: Backward-transfer heatmaps — continual obs change
    - **Exp 4**: Backward-transfer heatmaps — continual dynamics change
    - **Exp 5**: Learning curves on Task C for the three encoder conditions
    """)
    return


@app.cell
def _():
    import json
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    import warnings
    warnings.filterwarnings("ignore")

    RESULTS_DIR = "./results"

    CONDITION_COLORS = {
        "scratch":         "#4C72B0",
        "frozen_transfer": "#DD8452",
        "finetune":        "#55A868",
        "random_frozen":   "#C44E52",
        "frozen_aug":      "#8172B2",
        "frozen_from_A":   "#DD8452",
        "updated_on_B":    "#55A868",
        "scratch_C":       "#4C72B0",
        "frozen":          "#DD8452",
        "plastic":         "#55A868",
    }

    CONDITION_LABELS = {
        "scratch":         "Scratch",
        "frozen_transfer": "Frozen Transfer",
        "finetune":        "Fine-tune",
        "random_frozen":   "Random Frozen",
        "frozen_aug":      "Frozen Aug. Enc.",
        "frozen_from_A":   "Frozen from A",
        "updated_on_B":    "Updated on B→Frozen",
        "scratch_C":       "Scratch (C only)",
        "frozen":          "Frozen Enc. (after T1)",
        "plastic":         "Plastic Enc. + Policy",
        "scratch":         "Scratch per Task",
    }

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    def results_exist(*paths):
        return all(os.path.exists(p) for p in paths)

    def compute_aulc(curve):
        if not curve:
            return 0.0
        if len(curve) == 1:
            return float(curve[0][1])
        steps   = np.array([s for s, _ in curve], dtype=float)
        rewards = np.array([r for _, r in curve], dtype=float)
        return float(np.trapz(rewards, steps) / (steps[-1] - steps[0]))

    return (
        CONDITION_COLORS, CONDITION_LABELS, RESULTS_DIR,
        compute_aulc, gridspec, json, load_json, np, os, plt,
        results_exist, warnings,
    )


# ── Experiment 1 ─────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 1: Representation Transfer Under Observation Change
    """)
    return


@app.cell
def _(
    CONDITION_COLORS, CONDITION_LABELS, RESULTS_DIR,
    load_json, np, plt, results_exist,
):
    _aulc_path   = f"{RESULTS_DIR}/exp1/aulc.json"
    _curves_path = f"{RESULTS_DIR}/exp1/curves.json"

    if not results_exist(_aulc_path, _curves_path):
        print("Exp 1 results not found. Run train_experiments.py first.")
    else:
        _aulc   = load_json(_aulc_path)
        _curves = load_json(_curves_path)

        _conditions = ["scratch", "frozen_transfer", "finetune", "random_frozen"]
        _targets    = list(_aulc.keys())
        _n_t        = len(_targets)
        _n_c        = len(_conditions)
        _x          = np.arange(_n_t)
        _width      = 0.18

        # ── Figure 1a: AULC grouped bar chart ────────────────────────────────
        fig1a, ax1a = plt.subplots(figsize=(14, 5))
        for i, cond in enumerate(_conditions):
            _vals = [_aulc[t].get(cond, 0.0) for t in _targets]
            ax1a.bar(
                _x + (i - (_n_c - 1) / 2) * _width, _vals,
                _width, label=CONDITION_LABELS.get(cond, cond),
                color=CONDITION_COLORS.get(cond, None),
            )
        ax1a.set_xticks(_x)
        ax1a.set_xticklabels(_targets, rotation=30, ha="right")
        ax1a.set_ylabel("AULC (normalised)")
        ax1a.set_title("Exp 1 — Area Under Learning Curve by Target Env & Condition")
        ax1a.legend(loc="upper right")
        ax1a.axhline(0, color="gray", linewidth=0.5)
        plt.tight_layout()
        plt.show()

        # ── Figure 1b: Learning curves for a subset of targets ────────────────
        _show_targets = _targets[:4]  # first 4 for readability
        fig1b, axes1b = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
        for ax, tname in zip(axes1b.flat, _show_targets):
            for cond in _conditions:
                _curve = _curves[tname].get(cond, [])
                if _curve:
                    _steps   = [s for s, _ in _curve]
                    _rewards = [r for _, r in _curve]
                    ax.plot(
                        _steps, _rewards,
                        label=CONDITION_LABELS.get(cond, cond),
                        color=CONDITION_COLORS.get(cond, None),
                    )
            ax.set_title(tname)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean reward")
            ax.legend(fontsize=7)
            ax.axhline(0, color="gray", linewidth=0.4)
        plt.suptitle("Exp 1 — Learning Curves (subset of targets)")
        plt.tight_layout()
        plt.show()

    return


# ── Experiment 2 ─────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 2: Robustness from Augmentation
    """)
    return


@app.cell
def _(
    CONDITION_COLORS, CONDITION_LABELS, RESULTS_DIR,
    load_json, np, plt, results_exist,
):
    _res_path = f"{RESULTS_DIR}/exp2/results.json"

    if not results_exist(_res_path):
        print("Exp 2 results not found. Run train_experiments.py first.")
    else:
        _res = load_json(_res_path)
        _regimes = list(_res.keys())      # within_family, cross_family

        fig2, axes2 = plt.subplots(1, len(_regimes), figsize=(14, 5))
        if len(_regimes) == 1:
            axes2 = [axes2]

        for ax, regime in zip(axes2, _regimes):
            _targets  = list(_res[regime].keys())
            _conds    = list(next(iter(_res[regime].values())).keys())
            _x        = np.arange(len(_targets))
            _width    = 0.3

            for i, cond in enumerate(_conds):
                _vals = [_res[regime][t][cond].get("aulc", 0.0) for t in _targets]
                ax.bar(
                    _x + (i - (len(_conds) - 1) / 2) * _width,
                    _vals, _width,
                    label=CONDITION_LABELS.get(cond, cond),
                    color=CONDITION_COLORS.get(cond, None),
                )
            ax.set_xticks(_x)
            ax.set_xticklabels(_targets, rotation=20, ha="right")
            ax.set_ylabel("AULC (normalised)")
            ax.set_title(regime.replace("_", " ").title())
            ax.legend()
            ax.axhline(0, color="gray", linewidth=0.5)

        # Learning curves for each regime / target / condition
        fig2b, axes2b = plt.subplots(len(_regimes), max(len(list(_res[r].keys())) for r in _regimes),
                                     figsize=(14, 5 * len(_regimes)), squeeze=False)
        for row, regime in enumerate(_regimes):
            _targets = list(_res[regime].keys())
            for col, tname in enumerate(_targets):
                ax = axes2b[row][col]
                for cond, cdata in _res[regime][tname].items():
                    _curve = cdata.get("curve", [])
                    if _curve:
                        ax.plot(
                            [s for s, _ in _curve], [r for _, r in _curve],
                            label=CONDITION_LABELS.get(cond, cond),
                            color=CONDITION_COLORS.get(cond, None),
                        )
                ax.set_title(f"{regime} / {tname}")
                ax.set_xlabel("Timesteps")
                ax.set_ylabel("Mean reward")
                ax.legend(fontsize=7)
            # hide unused subplots in this row
            for col in range(len(_targets), axes2b.shape[1]):
                axes2b[row][col].set_visible(False)

        plt.figure(fig2.number)
        plt.suptitle("Exp 2 — AULC: Frozen Aug. Encoder vs Scratch")
        plt.tight_layout()
        plt.show()

        plt.figure(fig2b.number)
        plt.suptitle("Exp 2 — Learning Curves")
        plt.tight_layout()
        plt.show()

    return


# ── Experiment 3 ─────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 3: Continual Learning Under Observation Change

    Each heatmap shows performance on task *j* (column) after training on tasks 0..i (row).
    NaN = task not yet encountered.
    """)
    return


@app.cell
def _(CONDITION_COLORS, CONDITION_LABELS, RESULTS_DIR, load_json, np, plt, results_exist):
    _res_path  = f"{RESULTS_DIR}/exp3/results.json"
    _meta_path = f"{RESULTS_DIR}/exp3/meta.json"

    if not results_exist(_res_path, _meta_path):
        print("Exp 3 results not found. Run train_experiments.py first.")
    else:
        _res  = load_json(_res_path)
        _meta = load_json(_meta_path)
        _task_names = _meta["task_names"]
        _conditions = list(_res.keys())
        _n = len(_task_names)

        fig3, axes3 = plt.subplots(1, len(_conditions), figsize=(5 * len(_conditions), 4))
        if len(_conditions) == 1:
            axes3 = [axes3]

        for ax, cond in zip(axes3, _conditions):
            _mat = np.array(_res[cond])
            _masked = np.ma.masked_invalid(_mat)
            _vmin = np.nanmin(_mat[~np.isnan(_mat)]) if not np.all(np.isnan(_mat)) else 0
            _vmax = np.nanmax(_mat[~np.isnan(_mat)]) if not np.all(np.isnan(_mat)) else 1
            _im = ax.imshow(_masked, aspect="auto", cmap="RdYlGn",
                            vmin=_vmin, vmax=_vmax)
            ax.set_xticks(range(_n))
            ax.set_yticks(range(_n))
            ax.set_xticklabels(_task_names, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels([f"After T{i}: {n}" for i, n in enumerate(_task_names)],
                               fontsize=7)
            ax.set_xlabel("Eval task")
            ax.set_ylabel("Trained up to")
            ax.set_title(CONDITION_LABELS.get(cond, cond))
            plt.colorbar(_im, ax=ax, shrink=0.7, label="Mean reward")
            # Annotate cells
            for i in range(_n):
                for j in range(_n):
                    if not np.isnan(_mat[i, j]):
                        ax.text(j, i, f"{_mat[i,j]:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="black" if _mat[i, j] > (_vmin + _vmax) / 2 else "white")

        plt.suptitle("Exp 3 — Backward Transfer Matrix (Obs. Change)")
        plt.tight_layout()
        plt.show()

        # ── Summary: forgetting & forward transfer ────────────────────────────
        fig3s, axes3s = plt.subplots(1, 2, figsize=(12, 4))

        ax_fwd, ax_bwd = axes3s
        for cond in _conditions:
            _mat = np.array(_res[cond])
            # Forward transfer: perf on task j right after first training on j (diagonal)
            _diag = [_mat[i, i] for i in range(_n)]
            ax_fwd.plot(range(_n), _diag, marker="o",
                        label=CONDITION_LABELS.get(cond, cond),
                        color=CONDITION_COLORS.get(cond, None))
            # Forgetting: drop in performance on task 0 over time
            _forg = [_mat[i, 0] for i in range(_n)]
            ax_bwd.plot(range(_n), _forg, marker="s",
                        label=CONDITION_LABELS.get(cond, cond),
                        color=CONDITION_COLORS.get(cond, None))

        ax_fwd.set_xlabel("Task index")
        ax_fwd.set_ylabel("Mean reward (diagonal)")
        ax_fwd.set_title("Forward perf. on each new task")
        ax_fwd.set_xticks(range(_n))
        ax_fwd.set_xticklabels(_task_names, rotation=20, ha="right")
        ax_fwd.legend()

        ax_bwd.set_xlabel("Task trained up to")
        ax_bwd.set_ylabel("Mean reward on Task 0")
        ax_bwd.set_title("Forgetting of Task 0 over time")
        ax_bwd.set_xticks(range(_n))
        ax_bwd.set_xticklabels([f"After T{i}" for i in range(_n)], rotation=20, ha="right")
        ax_bwd.legend()

        plt.suptitle("Exp 3 — Forward Transfer & Forgetting")
        plt.tight_layout()
        plt.show()

    return


# ── Experiment 4 ─────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 4: Continual Learning Under Dynamics Change

    Same structure as Exp 3 but with fixed observations and changing goal positions.
    """)
    return


@app.cell
def _(CONDITION_COLORS, CONDITION_LABELS, RESULTS_DIR, load_json, np, plt, results_exist):
    _res_path  = f"{RESULTS_DIR}/exp4/results.json"
    _meta_path = f"{RESULTS_DIR}/exp4/meta.json"

    if not results_exist(_res_path, _meta_path):
        print("Exp 4 results not found. Run train_experiments.py first.")
    else:
        _res  = load_json(_res_path)
        _meta = load_json(_meta_path)
        _task_names = _meta["task_names"]
        _conditions = list(_res.keys())
        _n = len(_task_names)

        # Heatmaps
        fig4, axes4 = plt.subplots(1, len(_conditions), figsize=(5 * len(_conditions), 4))
        if len(_conditions) == 1:
            axes4 = [axes4]

        for ax, cond in zip(axes4, _conditions):
            _mat = np.array(_res[cond])
            _masked = np.ma.masked_invalid(_mat)
            _vmin = np.nanmin(_mat[~np.isnan(_mat)]) if not np.all(np.isnan(_mat)) else 0
            _vmax = np.nanmax(_mat[~np.isnan(_mat)]) if not np.all(np.isnan(_mat)) else 1
            _im = ax.imshow(_masked, aspect="auto", cmap="RdYlGn",
                            vmin=_vmin, vmax=_vmax)
            ax.set_xticks(range(_n))
            ax.set_yticks(range(_n))
            ax.set_xticklabels(_task_names, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels([f"After T{i}: {n}" for i, n in enumerate(_task_names)],
                               fontsize=7)
            ax.set_xlabel("Eval task")
            ax.set_ylabel("Trained up to")
            ax.set_title(CONDITION_LABELS.get(cond, cond))
            plt.colorbar(_im, ax=ax, shrink=0.7, label="Mean reward")
            for i in range(_n):
                for j in range(_n):
                    if not np.isnan(_mat[i, j]):
                        ax.text(j, i, f"{_mat[i,j]:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="black" if _mat[i, j] > (_vmin + _vmax) / 2 else "white")

        plt.suptitle("Exp 4 — Backward Transfer Matrix (Dynamics Change)")
        plt.tight_layout()
        plt.show()

        # Summary: forward + forgetting
        fig4s, axes4s = plt.subplots(1, 2, figsize=(12, 4))
        ax_fwd4, ax_bwd4 = axes4s
        for cond in _conditions:
            _mat = np.array(_res[cond])
            ax_fwd4.plot(range(_n), [_mat[i, i] for i in range(_n)],
                         marker="o", label=CONDITION_LABELS.get(cond, cond),
                         color=CONDITION_COLORS.get(cond, None))
            ax_bwd4.plot(range(_n), [_mat[i, 0] for i in range(_n)],
                         marker="s", label=CONDITION_LABELS.get(cond, cond),
                         color=CONDITION_COLORS.get(cond, None))

        ax_fwd4.set_xlabel("Task index")
        ax_fwd4.set_ylabel("Mean reward (diagonal)")
        ax_fwd4.set_title("Forward perf. on each new task")
        ax_fwd4.set_xticks(range(_n))
        ax_fwd4.set_xticklabels(_task_names, rotation=20, ha="right")
        ax_fwd4.legend()

        ax_bwd4.set_xlabel("Task trained up to")
        ax_bwd4.set_ylabel("Mean reward on Task 0")
        ax_bwd4.set_title("Forgetting of Task 0 over time")
        ax_bwd4.set_xticks(range(_n))
        ax_bwd4.set_xticklabels([f"After T{i}" for i in range(_n)], rotation=20, ha="right")
        ax_bwd4.legend()

        plt.suptitle("Exp 4 — Forward Transfer & Forgetting")
        plt.tight_layout()
        plt.show()

    return


# ── Experiment 5 ─────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 5: Is Updating the Representation Actually Useful?

    Learning curves on Task C (new obstacle layout) for three encoder conditions.
    If `updated_on_B` learns faster than `frozen_from_A`, the encoder acquired
    reusable structure from Task B.
    """)
    return


@app.cell
def _(
    CONDITION_COLORS, CONDITION_LABELS, RESULTS_DIR,
    compute_aulc, load_json, np, plt, results_exist,
):
    _curves_path = f"{RESULTS_DIR}/exp5/curves.json"
    _aulc_path   = f"{RESULTS_DIR}/exp5/aulc.json"

    if not results_exist(_curves_path, _aulc_path):
        print("Exp 5 results not found. Run train_experiments.py first.")
    else:
        _curves = load_json(_curves_path)
        _aulc   = load_json(_aulc_path)

        _conditions = list(_curves.keys())

        fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))

        # Left: learning curves on Task C
        ax5_lc = axes5[0]
        for cond in _conditions:
            _curve = _curves[cond]
            if _curve:
                _steps   = [s for s, _ in _curve]
                _rewards = [r for _, r in _curve]
                ax5_lc.plot(
                    _steps, _rewards,
                    label=CONDITION_LABELS.get(cond, cond),
                    color=CONDITION_COLORS.get(cond, None),
                    linewidth=2,
                )
        ax5_lc.set_xlabel("Timesteps on Task C")
        ax5_lc.set_ylabel("Mean reward")
        ax5_lc.set_title("Exp 5 — Learning Curves on Task C")
        ax5_lc.legend()
        ax5_lc.axhline(0, color="gray", linewidth=0.5)

        # Right: AULC bar chart
        ax5_bar = axes5[1]
        _x = np.arange(len(_conditions))
        _vals = [_aulc.get(c, 0.0) for c in _conditions]
        _bars = ax5_bar.bar(
            _x, _vals,
            color=[CONDITION_COLORS.get(c, "steelblue") for c in _conditions],
        )
        ax5_bar.set_xticks(_x)
        ax5_bar.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in _conditions],
            rotation=15, ha="right",
        )
        ax5_bar.set_ylabel("AULC (normalised)")
        ax5_bar.set_title("Exp 5 — AULC on Task C")
        ax5_bar.axhline(0, color="gray", linewidth=0.5)
        for bar, val in zip(_bars, _vals):
            ax5_bar.text(bar.get_x() + bar.get_width() / 2,
                         val + 0.005, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=9)

        plt.suptitle("Exp 5 — Representation Update Utility")
        plt.tight_layout()
        plt.show()

        # Print summary
        print("\nExp 5 AULC summary:")
        for cond in _conditions:
            print(f"  {CONDITION_LABELS.get(cond, cond):35s}: {_aulc.get(cond, 0.0):.4f}")

        _base = _aulc.get("frozen_from_A", None)
        _upd  = _aulc.get("updated_on_B", None)
        if _base is not None and _upd is not None:
            _delta = _upd - _base
            print(f"\n  Delta (updated_on_B - frozen_from_A): {_delta:+.4f}")
            if _delta > 0:
                print("  → Representation updating acquired reusable structure.")
            else:
                print("  → No evidence of reusable structure from Task B update.")

    return


if __name__ == "__main__":
    app.run()
