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
    # Phase 1: Representation Utility under Distribution Shift — Training

    Trains models for all 5 experiments and saves results to `./results/`.

    - **Exp 1**: Representation transfer under observation change (4 conditions × 8 targets)
    - **Exp 2**: Robustness from augmentation (within-family + cross-family)
    - **Exp 3**: Continual learning under observation change
    - **Exp 4**: Continual learning under dynamics change
    - **Exp 5**: Is updating the representation actually useful?

    Run cells top-to-bottom. Each experiment saves independently so you can resume.
    """)
    return


@app.cell
def _():
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn
    import os
    import json
    import warnings
    import cv2
    warnings.filterwarnings("ignore")

    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    import minigrid
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Goal, Wall
    from minigrid.minigrid_env import MiniGridEnv
    from minigrid.core.mission import MissionSpace

    return (
        BaseCallback,
        BaseFeaturesExtractor,
        DummyVecEnv,
        Goal,
        Grid,
        ImgObsWrapper,
        MiniGridEnv,
        MissionSpace,
        Monitor,
        PPO,
        RGBImgObsWrapper,
        Wall,
        cv2,
        evaluate_policy,
        gym,
        json,
        nn,
        np,
        os,
        torch,
    )


@app.cell
def _(BaseFeaturesExtractor, gym, nn, torch):
    class MinigridFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.Space, features_dim: int = 128) -> None:
            super().__init__(observation_space, features_dim)
            n_ch = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_ch, 16, (2, 2)), nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)), nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flat = self.cnn(
                    torch.as_tensor(observation_space.sample()[None]).float()
                ).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(obs))

    return (MinigridFeaturesExtractor,)


@app.cell
def _(MinigridFeaturesExtractor):
    POLICY_KWARGS = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    BASE_PPO = dict(
        policy="CnnPolicy",
        policy_kwargs=POLICY_KWARGS,
        learning_rate=1e-4,
        n_epochs=4,
        target_kl=0.02,
        clip_range=0.2,
        clip_range_vf=0.2,
        max_grad_norm=0.5,
        seed=0,
        device="cpu",
        verbose=1,
    )
    return (BASE_PPO,)


@app.cell
def _(ImgObsWrapper, RGBImgObsWrapper, cv2, gym, np):
    # ── Observation wrappers ────────────────────────────────────────────────────

    class AffineTransform(gym.ObservationWrapper):
        def __init__(self, env, angle=15.0, scale=1.0, translate=(0, 0)):
            super().__init__(env)
            self.angle, self.scale, self.translate = angle, scale, translate

        def observation(self, obs):
            h, w = obs.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), self.angle, self.scale)
            M[0, 2] += self.translate[0]
            M[1, 2] += self.translate[1]
            return cv2.warpAffine(obs, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

    class CyclicTranslation(gym.ObservationWrapper):
        def __init__(self, env, shift_x=0, shift_y=0):
            super().__init__(env)
            self.sx, self.sy = shift_x, shift_y

        def observation(self, obs):
            return np.roll(np.roll(obs, self.sy, axis=0), self.sx, axis=1).copy()

    class GlobalColorShift(gym.ObservationWrapper):
        def __init__(self, env, shift=(50, 50, 50)):
            super().__init__(env)
            self.shift = np.array(shift, dtype=np.int32)

        def observation(self, obs):
            return ((obs.astype(np.int32) + self.shift) % 256).astype(np.uint8)

    class SpriteTextureSwap(gym.ObservationWrapper):
        def __init__(self, env, swap_agent=False, swap_obstacles=False,
                     swap_goal=False, swap_floor=False):
            super().__init__(env)
            self.swap_agent = swap_agent
            self.swap_obstacles = swap_obstacles
            self.swap_goal = swap_goal
            self.swap_floor = swap_floor

        @staticmethod
        def _draw_key(tile, agent_dir):
            import math
            from minigrid.core.constants import COLORS
            from minigrid.utils.rendering import (
                fill_coords, point_in_rect, point_in_circle, rotate_fn)
            c = COLORS["red"]
            theta = {0: -math.pi / 2, 1: 0, 2: math.pi / 2, 3: math.pi}[agent_dir]
            def rot(fn): return rotate_fn(fn, 0.5, 0.5, theta)
            fill_coords(tile, rot(point_in_rect(0.50, 0.63, 0.31, 0.88)), c)
            fill_coords(tile, rot(point_in_rect(0.38, 0.50, 0.59, 0.66)), c)
            fill_coords(tile, rot(point_in_rect(0.38, 0.50, 0.81, 0.88)), c)
            fill_coords(tile, rot(point_in_circle(cx=0.56, cy=0.28, r=0.190)), c)
            fill_coords(tile, rot(point_in_circle(cx=0.56, cy=0.28, r=0.064)), (0, 0, 0))

        @staticmethod
        def _draw_grass(tile):
            from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_triangle
            fill_coords(tile, point_in_rect(0, 1, 0, 1), (100, 70, 40))
            for a, b, tip, col in [
                ((0.05, 0.95), (0.22, 0.95), (0.13, 0.20), (0, 160, 0)),
                ((0.20, 0.95), (0.42, 0.95), (0.30, 0.10), (0, 190, 0)),
                ((0.38, 0.95), (0.58, 0.95), (0.50, 0.30), (0, 150, 0)),
                ((0.55, 0.95), (0.75, 0.95), (0.62, 0.05), (34, 200, 34)),
                ((0.72, 0.95), (0.92, 0.95), (0.82, 0.25), (0, 170, 0)),
            ]:
                fill_coords(tile, point_in_triangle(a, b, tip), col)

        @staticmethod
        def _draw_chest(tile):
            from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle
            fill_coords(tile, point_in_rect(0.10, 0.90, 0.35, 0.90), (110, 60, 20))
            fill_coords(tile, point_in_rect(0.10, 0.90, 0.25, 0.45), (140, 80, 30))
            fill_coords(tile, point_in_circle(cx=0.50, cy=0.30, r=0.22), (140, 80, 30))
            fill_coords(tile, point_in_rect(0.10, 0.90, 0.44, 0.52), (180, 160, 50))
            fill_coords(tile, point_in_rect(0.42, 0.58, 0.25, 0.90), (180, 160, 50))
            fill_coords(tile, point_in_circle(cx=0.50, cy=0.65, r=0.09), (220, 190, 60))
            fill_coords(tile, point_in_circle(cx=0.50, cy=0.65, r=0.04), (0, 0, 0))

        @staticmethod
        def _draw_sand(tile):
            from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle
            fill_coords(tile, point_in_rect(0, 1, 0, 1), (210, 180, 120))
            for cx, cy, r in [
                (0.20, 0.25, 0.08), (0.70, 0.15, 0.06), (0.45, 0.55, 0.07),
                (0.15, 0.75, 0.05), (0.80, 0.70, 0.07), (0.55, 0.85, 0.06),
            ]:
                fill_coords(tile, point_in_circle(cx, cy, r), (185, 155, 95))
            for cx, cy, r in [
                (0.35, 0.15, 0.05), (0.60, 0.40, 0.06),
                (0.25, 0.60, 0.04), (0.85, 0.45, 0.05),
            ]:
                fill_coords(tile, point_in_circle(cx, cy, r), (230, 205, 150))

        def _tile_array(self, obs, gx, gy):
            base = self.unwrapped
            gh, gw = base.grid.height, base.grid.width
            ih, iw = obs.shape[:2]
            x0 = int(round(gx * iw / gw))
            x1 = min(int(round((gx + 1) * iw / gw)), iw)
            y0 = int(round(gy * ih / gh))
            y1 = min(int(round((gy + 1) * ih / gh)), ih)
            return np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8), slice(y0, y1), slice(x0, x1)

        def observation(self, obs):
            from minigrid.core.world_object import Wall as _Wall, Goal as _Goal
            out = obs.copy()
            base = self.unwrapped
            gh, gw = base.grid.height, base.grid.width
            ax, ay = base.agent_pos
            for gy in range(1, gh - 1):
                for gx in range(1, gw - 1):
                    obj = base.grid.get(gx, gy)
                    if self.swap_obstacles and isinstance(obj, _Wall):
                        t, sy, sx = self._tile_array(obs, gx, gy)
                        self._draw_grass(t); out[sy, sx] = t
                    elif self.swap_goal and isinstance(obj, _Goal):
                        t, sy, sx = self._tile_array(obs, gx, gy)
                        self._draw_chest(t); out[sy, sx] = t
                    elif self.swap_floor and obj is None and (gx, gy) != (ax, ay):
                        t, sy, sx = self._tile_array(obs, gx, gy)
                        self._draw_sand(t); out[sy, sx] = t
            if self.swap_agent:
                t, sy, sx = self._tile_array(obs, ax, ay)
                self._draw_key(t, base.agent_dir); out[sy, sx] = t
            return out

    class RandomAugWrapper(gym.ObservationWrapper):
        """Randomly applies one augmentation function per step from a list."""
        def __init__(self, env, aug_fns):
            super().__init__(env)
            self.aug_fns = aug_fns

        def observation(self, obs):
            fn = self.aug_fns[np.random.randint(len(self.aug_fns))]
            return fn(obs)

    def make_base_env():
        env = gym.make("MiniGrid-Empty-5x5-v0")
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    return (
        AffineTransform,
        CyclicTranslation,
        GlobalColorShift,
        RandomAugWrapper,
        SpriteTextureSwap,
        make_base_env,
    )


@app.cell
def _(
    Goal,
    Grid,
    ImgObsWrapper,
    MiniGridEnv,
    MissionSpace,
    RGBImgObsWrapper,
    Wall,
):
    # ── Custom environments for dynamics experiments ────────────────────────────

    class EmptyGridCustomGoal(MiniGridEnv):
        """5×5 empty grid with a configurable goal position (dynamics change)."""
        def __init__(self, goal_pos=(3, 3), **kwargs):
            self.goal_pos = goal_pos
            super().__init__(
                mission_space=MissionSpace(mission_func=lambda: "get to the goal"),
                grid_size=5, max_steps=4 * 5 * 5, **kwargs,
            )

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.put_obj(Goal(), *self.goal_pos)
            self.place_agent()

    class ObstacleGridEnv(MiniGridEnv):
        """5×5 grid with fixed interior wall obstacles and goal at (3,3)."""
        def __init__(self, obstacle_positions=(), **kwargs):
            self.obstacle_positions = list(obstacle_positions)
            super().__init__(
                mission_space=MissionSpace(mission_func=lambda: "get to the goal"),
                grid_size=5, max_steps=4 * 5 * 5, **kwargs,
            )

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.put_obj(Goal(), 3, 3)
            for pos in self.obstacle_positions:
                self.put_obj(Wall(), *pos)
            self.place_agent()

    def make_goal_env(goal_pos):
        env = EmptyGridCustomGoal(goal_pos=goal_pos)
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    def make_obstacle_env(obstacle_positions):
        env = ObstacleGridEnv(obstacle_positions=obstacle_positions)
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    return make_goal_env, make_obstacle_env


@app.cell
def _(BaseCallback, PPO, evaluate_policy, json, np, os):
    # ── Shared helpers ──────────────────────────────────────────────────────────

    class LearningCurveCallback(BaseCallback):
        """Periodically evaluates the policy and records (timestep, mean_reward)."""
        def __init__(self, eval_env_fn, eval_freq=5000, n_eval_episodes=10):
            super().__init__(verbose=0)
            self.eval_env_fn = eval_env_fn
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.curve = []  # list of [timestep, mean_reward]

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                eval_env = self.eval_env_fn()
                mean_r, _ = evaluate_policy(
                    self.model, eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=False,
                )
                eval_env.close()
                self.curve.append([int(self.num_timesteps), float(mean_r)])
            return True

    def make_ppo(env, base_ppo):
        return PPO(env=env, **base_ppo)

    def freeze_encoder(model):
        for p in model.policy.features_extractor.parameters():
            p.requires_grad_(False)
        return model

    def transfer_encoder(src, dst):
        """Copy only the features extractor weights."""
        dst.policy.features_extractor.load_state_dict(
            src.policy.features_extractor.state_dict()
        )
        return dst

    def transfer_all_policy(src, dst):
        """Copy all policy weights (encoder + MLP + heads)."""
        dst.policy.load_state_dict(src.policy.state_dict())
        return dst

    def train_with_curve(savedir,model, total_steps, eval_env_fn,
                         eval_freq=5000, n_eval_episodes=10):
        cb = LearningCurveCallback(eval_env_fn, eval_freq, n_eval_episodes)
        model.learn(total_steps, callback=cb, reset_num_timesteps=True)
        model.save(savedir)
        return cb.curve

    def compute_aulc(curve):
        """Trapezoidal area under learning curve, normalised by step range."""
        if not curve:
            return 0.0
        if len(curve) == 1:
            return float(curve[0][1])
        steps = np.array([s for s, _ in curve], dtype=float)
        rewards = np.array([r for _, r in curve], dtype=float)
        return float(np.trapezoid(rewards, steps) / (steps[-1] - steps[0]))

    def save_json(path, data):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    return (
        compute_aulc,
        freeze_encoder,
        load_json,
        make_ppo,
        save_json,
        train_with_curve,
        transfer_all_policy,
        transfer_encoder,
    )


@app.cell
def _(os):
    RESULTS_DIR = "./results"
    for _d in [f"{RESULTS_DIR}/exp{i}" for i in range(1, 6)]:
        os.makedirs(_d, exist_ok=True)
    print(f"Results directory: {RESULTS_DIR}")
    return (RESULTS_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 1: Representation Transfer Under Observation Change

    **Protocol**
    1. Train encoder + policy on source (baseline MiniGrid-Empty-5x5).
    2. For each target env (same MDP, changed observations), compare:
       - `scratch` – fresh model trained from random init
       - `frozen_transfer` – source encoder frozen, new policy head
       - `finetune` – all source weights, all params updated
       - `random_frozen` – random encoder frozen, new policy head
    3. Metric: area under learning curve (AULC).
    """)
    return


@app.cell
def _(
    AffineTransform,
    CyclicTranslation,
    GlobalColorShift,
    Monitor,
    SpriteTextureSwap,
    make_base_env,
):
    SOURCE_STEPS = 100_000
    TARGET_STEPS = 50_000
    EVAL_FREQ    = 5_000
    N_EVAL_EPS   = 10

    # ── Target env factories ──────────────────────────────────────────────────
    targets = {
        "rot_30":      lambda: Monitor(AffineTransform(make_base_env(), angle=30)),
        "trans_x":     lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8)),
        "trans_y":     lambda: Monitor(CyclicTranslation(make_base_env(), shift_y=8)),
        "trans_xy":    lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8, shift_y=8)),
        "color_shift": lambda: Monitor(GlobalColorShift(make_base_env(), shift=(50, 50, 50))),
        "swap_goal":   lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_goal=True)),
        "swap_agent":  lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_agent=True)),
        "swap_floor":  lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_floor=True)),
    }
    return EVAL_FREQ, N_EVAL_EPS, SOURCE_STEPS, TARGET_STEPS, targets


@app.cell
def _(
    BASE_PPO,
    Monitor,
    PPO,
    RESULTS_DIR,
    SOURCE_STEPS,
    make_base_env,
    make_ppo,
    os,
):
    # ── Source training ───────────────────────────────────────────────────────
    _src_path = f"{RESULTS_DIR}/exp1/source_model.zip"
    if os.path.exists(_src_path):
        print(f"=== Exp 1: loading existing source model from {_src_path} ===")
        _src_env = Monitor(make_base_env())
        exp1_source = PPO.load(_src_path, env=_src_env)
    else:
        print("=== Exp 1: training source model ===")
        _src_env = Monitor(make_base_env())
        exp1_source = make_ppo(_src_env, BASE_PPO)
        exp1_source.learn(SOURCE_STEPS)
        exp1_source.save(f"{RESULTS_DIR}/exp1/source_model")
        print("Source model saved.")
    return (exp1_source,)


@app.cell
def _(
    BASE_PPO,
    EVAL_FREQ,
    N_EVAL_EPS,
    PPO,
    RESULTS_DIR,
    TARGET_STEPS,
    compute_aulc,
    exp1_source,
    freeze_encoder,
    load_json,
    make_ppo,
    os,
    save_json,
    targets,
    train_with_curve,
    transfer_all_policy,
    transfer_encoder,
):

    # Check if all results are already saved
    _curves_path = f"{RESULTS_DIR}/exp1/curves.json"
    _aulc_path = f"{RESULTS_DIR}/exp1/aulc.json"
    if os.path.exists(_curves_path) and os.path.exists(_aulc_path):
        print("=== Exp 1: loading existing results ===")
        exp1_curves = load_json(_curves_path)
        exp1_aulc = load_json(_aulc_path)
        print(f"  Loaded curves for targets: {list(exp1_curves.keys())}")
    else:
        exp1_curves = {}
        exp1_aulc   = {}

        for _tname, _tfn in targets.items():
            print(f"\n  target: {_tname}")
            exp1_curves[_tname] = {}

            _modes = {
                "scratch":         lambda: make_ppo(_tfn(), BASE_PPO),
                "frozen_transfer": lambda: freeze_encoder(transfer_encoder(exp1_source, make_ppo(_tfn(), BASE_PPO))),
                "finetune":        lambda: transfer_all_policy(exp1_source, make_ppo(_tfn(), BASE_PPO)),
                "random_frozen":   lambda: freeze_encoder(make_ppo(_tfn(), BASE_PPO)),
            }

            for _mode, _make_model in _modes.items():
                _savedir = f"{RESULTS_DIR}/exp1/{_tname}_{_mode}"
                if os.path.exists(f"{_savedir}.zip"):
                    print(f"    {_mode}: loading existing model from {_savedir}.zip")
                    _m = PPO.load(_savedir, env=_tfn())
                    exp1_curves[_tname][_mode] = []  # curve not available for loaded models
                else:
                    print(f"    {_mode}: training")
                    _m = _make_model()
                    exp1_curves[_tname][_mode] = train_with_curve(
                        _savedir, _m, TARGET_STEPS, _tfn, EVAL_FREQ, N_EVAL_EPS)

            exp1_aulc[_tname] = {
                cond: compute_aulc(curve)
                for cond, curve in exp1_curves[_tname].items()
            }
            print(f"    AULC: {exp1_aulc[_tname]}")

        save_json(_curves_path, exp1_curves)
        save_json(_aulc_path, exp1_aulc)
        print("\nExp 1 complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 2: Robustness from Augmentation

    Two regimes:
    - **within-family**: train on color shifts {red, green}, test on {blue, yellow}
    - **cross-family**: train on {color shifts + sprite swaps}, test on {rotation, translation}

    Baseline for each regime: scratch on the held-out target.
    """)
    return


@app.cell
def _(
    AffineTransform,
    BASE_PPO,
    CyclicTranslation,
    GlobalColorShift,
    Monitor,
    PPO,
    RESULTS_DIR,
    RandomAugWrapper,
    SpriteTextureSwap,
    c,
    compute_aulc,
    freeze_encoder,
    load_json,
    make_base_env,
    make_ppo,
    os,
    save_json,
    train_with_curve,
    transfer_encoder,
):
    _results_path = f"{RESULTS_DIR}/exp2/results.json"
    if os.path.exists(_results_path):
        print("=== Exp 2: loading existing results ===")
        exp2_results = load_json(_results_path)
    else:
        _SOURCE_STEPS = 100_000
        _TARGET_STEPS = 50_000
        _EVAL_FREQ    = 5_000
        _N_EVAL_EPS   = 10

        exp2_results = {}

        # ── Within-family: color ─────────────────────────────────────────────────
        print("=== Exp 2: within-family (color) ===")

        def _make_wf_aug_env():
            base = make_base_env()
            return Monitor(RandomAugWrapper(base, [
                lambda o: ((o.astype("int32") + [50, 0, 0]) % 256).astype("uint8"),
                lambda o: ((o.astype("int32") + [0, 50, 0]) % 256).astype("uint8"),
            ]))

        _wf_model_path = f"{RESULTS_DIR}/exp2/within_family_aug_model.zip"
        if os.path.exists(_wf_model_path):
            print("  Loading existing within-family aug model")
            _aug_model = PPO.load(_wf_model_path, env=_make_wf_aug_env())
        else:
            _aug_env = _make_wf_aug_env()
            _aug_model = make_ppo(_aug_env, BASE_PPO)
            _aug_model.learn(_SOURCE_STEPS)
            _aug_model.save(f"{RESULTS_DIR}/exp2/within_family_aug_model")

        _wf_targets = {
            "blue_shift":   lambda: Monitor(GlobalColorShift(make_base_env(), shift=(0, 0, 50))),
            "yellow_shift": lambda: Monitor(GlobalColorShift(make_base_env(), shift=(50, 50, 0))),
        }

        exp2_results["within_family"] = {}
        for _tname, _tfn in _wf_targets.items():
            exp2_results["within_family"][_tname] = {}

            _m = make_ppo(_tfn(), BASE_PPO)
            transfer_encoder(_aug_model, _m)
            freeze_encoder(_m)
            exp2_results["within_family"][_tname]["frozen_aug"] = {
                "curve": train_with_curve(_m, _TARGET_STEPS, _tfn, _EVAL_FREQ, _N_EVAL_EPS)
            }

            _m = make_ppo(_tfn(), BASE_PPO)
            exp2_results["within_family"][_tname]["scratch"] = {
                "curve": train_with_curve(_m, _TARGET_STEPS, _tfn, _EVAL_FREQ, _N_EVAL_EPS)
            }

            for _cond in exp2_results["within_family"][_tname]:
                exp2_results["within_family"][_tname][_cond]["aulc"] = compute_aulc(
                    exp2_results["within_family"][_tname][_cond]["curve"]
                )
            print(f"  {_tname}: {{{c: v['aulc']:.2f} for c,v in exp2_results['within_family'][_tname].items()}}")

        # ── Cross-family: color+sprite → rotation+translation ────────────────────
        print("\n=== Exp 2: cross-family ===")

        def _make_cf_aug_env():
            base = make_base_env()
            wrappers_fns = [
                lambda o: ((o.astype("int32") + [50, 0, 0]) % 256).astype("uint8"),
                lambda o: ((o.astype("int32") + [0, 50, 0]) % 256).astype("uint8"),
            ]
            return Monitor(RandomAugWrapper(
                SpriteTextureSwap(base, swap_goal=True, swap_agent=True),
                wrappers_fns,
            ))

        _cf_model_path = f"{RESULTS_DIR}/exp2/cross_family_aug_model.zip"
        if os.path.exists(_cf_model_path):
            print("  Loading existing cross-family aug model")
            _cf_model = PPO.load(_cf_model_path, env=_make_cf_aug_env())
        else:
            _cf_env = _make_cf_aug_env()
            _cf_model = make_ppo(_cf_env, BASE_PPO)
            _cf_model.learn(_SOURCE_STEPS)
            _cf_model.save(f"{RESULTS_DIR}/exp2/cross_family_aug_model")

        _cf_targets = {
            "rot_30":   lambda: Monitor(AffineTransform(make_base_env(), angle=30)),
            "trans_x":  lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8)),
            "trans_xy": lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8, shift_y=8)),
        }

        exp2_results["cross_family"] = {}
        for _tname, _tfn in _cf_targets.items():
            exp2_results["cross_family"][_tname] = {}

            _m = make_ppo(_tfn(), BASE_PPO)
            transfer_encoder(_cf_model, _m)
            freeze_encoder(_m)
            exp2_results["cross_family"][_tname]["frozen_aug"] = {
                "curve": train_with_curve(_m, _TARGET_STEPS, _tfn, _EVAL_FREQ, _N_EVAL_EPS)
            }

            _m = make_ppo(_tfn(), BASE_PPO)
            exp2_results["cross_family"][_tname]["scratch"] = {
                "curve": train_with_curve(_m, _TARGET_STEPS, _tfn, _EVAL_FREQ, _N_EVAL_EPS)
            }

            for _cond in exp2_results["cross_family"][_tname]:
                exp2_results["cross_family"][_tname][_cond]["aulc"] = compute_aulc(
                    exp2_results["cross_family"][_tname][_cond]["curve"]
                )

        save_json(_results_path, exp2_results)
        print("Exp 2 complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 3: Continual Learning Under Observation Change

    Task sequence: baseline → color change → sprite change → rotation

    Conditions:
    - `frozen`: train full model on Task 1, freeze encoder, update only policy on Tasks 2–4
    - `plastic`: update encoder + policy throughout the sequence
    - `scratch`: fresh model trained independently on each task

    After each task, evaluate on all tasks seen so far → backward transfer matrix.
    """)
    return


@app.cell
def _(
    AffineTransform,
    BASE_PPO,
    DummyVecEnv,
    GlobalColorShift,
    Monitor,
    RESULTS_DIR,
    SpriteTextureSwap,
    evaluate_policy,
    freeze_encoder,
    load_json,
    make_base_env,
    make_ppo,
    np,
    os,
    save_json,
):
    _results_path = f"{RESULTS_DIR}/exp3/results.json"
    _meta_path = f"{RESULTS_DIR}/exp3/meta.json"
    if os.path.exists(_results_path) and os.path.exists(_meta_path):
        print("=== Exp 3: loading existing results ===")
        exp3_results = load_json(_results_path)
        exp3_meta = load_json(_meta_path)
    else:
        _STEPS_PER_TASK = 100_000
        _N_EVAL_EPS     = 20

        _task_seq = [
            ("baseline",     lambda: Monitor(make_base_env())),
            ("color_shift",  lambda: Monitor(GlobalColorShift(make_base_env(), shift=(50, 50, 50)))),
            ("sprite_swap",  lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_goal=True, swap_agent=True))),
            ("rotation",     lambda: Monitor(AffineTransform(make_base_env(), angle=30))),
        ]
        _n = len(_task_seq)

        def _run_condition(condition_name):
            matrix = np.full((_n, _n), np.nan)
            model = None
            encoder_frozen = False

            for i, (tname, env_fn) in enumerate(_task_seq):
                print(f"  [{condition_name}] task {i}: {tname}")

                if condition_name == "scratch":
                    train_env = env_fn()
                    model = make_ppo(train_env, BASE_PPO)
                    model.learn(_STEPS_PER_TASK, reset_num_timesteps=True)

                elif condition_name == "plastic":
                    train_env = env_fn()
                    if model is None:
                        model = make_ppo(train_env, BASE_PPO)
                    else:
                        model.set_env(DummyVecEnv([env_fn]))
                    model.learn(_STEPS_PER_TASK, reset_num_timesteps=False)

                elif condition_name == "frozen":
                    train_env = env_fn()
                    if model is None:
                        model = make_ppo(train_env, BASE_PPO)
                        model.learn(_STEPS_PER_TASK, reset_num_timesteps=True)
                    else:
                        if not encoder_frozen:
                            freeze_encoder(model)
                            encoder_frozen = True
                        model.set_env(DummyVecEnv([env_fn]))
                        model.learn(_STEPS_PER_TASK, reset_num_timesteps=False)

                # Evaluate on all tasks so far
                for j, (ename, eval_fn) in enumerate(_task_seq):
                    eval_env = eval_fn()
                    mean_r, _ = evaluate_policy(
                        model, eval_env, n_eval_episodes=_N_EVAL_EPS, deterministic=False)
                    matrix[i, j] = float(mean_r)
                    eval_env.close()

            return matrix.tolist()

        exp3_results = {}
        for _cond in ("scratch", "plastic", "frozen"):
            print(f"\n=== Exp 3: condition = {_cond} ===")
            exp3_results[_cond] = _run_condition(_cond)

        exp3_meta = {"task_names": [n for n, _ in _task_seq]}
        save_json(_results_path, exp3_results)
        save_json(_meta_path, exp3_meta)
        print("\nExp 3 complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 4: Continual Learning Under Dynamics Change

    Observations stay fixed (baseline visuals); only the goal position changes.

    Task sequence: goal@(3,3) → goal@(1,1) → goal@(3,1) → goal@(1,3)

    Same conditions and measurement as Experiment 3.
    """)
    return


@app.cell
def _(
    BASE_PPO,
    DummyVecEnv,
    Monitor,
    RESULTS_DIR,
    evaluate_policy,
    freeze_encoder,
    load_json,
    make_goal_env,
    make_ppo,
    np,
    os,
    save_json,
):
    _results_path = f"{RESULTS_DIR}/exp4/results.json"
    _meta_path = f"{RESULTS_DIR}/exp4/meta.json"
    if os.path.exists(_results_path) and os.path.exists(_meta_path):
        print("=== Exp 4: loading existing results ===")
        exp4_results = load_json(_results_path)
        exp4_meta = load_json(_meta_path)
    else:
        _STEPS_PER_TASK = 100_000
        _N_EVAL_EPS     = 20

        _task_seq4 = [
            ("goal_33", lambda: Monitor(make_goal_env((3, 3)))),
            ("goal_11", lambda: Monitor(make_goal_env((1, 1)))),
            ("goal_31", lambda: Monitor(make_goal_env((3, 1)))),
            ("goal_13", lambda: Monitor(make_goal_env((1, 3)))),
        ]
        _n4 = len(_task_seq4)

        def _run_condition4(condition_name):
            matrix = np.full((_n4, _n4), np.nan)
            model = None
            encoder_frozen = False

            for i, (tname, env_fn) in enumerate(_task_seq4):
                print(f"  [{condition_name}] task {i}: {tname}")

                if condition_name == "scratch":
                    train_env = env_fn()
                    model = make_ppo(train_env, BASE_PPO)
                    model.learn(_STEPS_PER_TASK, reset_num_timesteps=True)

                elif condition_name == "plastic":
                    train_env = env_fn()
                    if model is None:
                        model = make_ppo(train_env, BASE_PPO)
                    else:
                        model.set_env(DummyVecEnv([env_fn]))
                    model.learn(_STEPS_PER_TASK, reset_num_timesteps=False)

                elif condition_name == "frozen":
                    train_env = env_fn()
                    if model is None:
                        model = make_ppo(train_env, BASE_PPO)
                        model.learn(_STEPS_PER_TASK, reset_num_timesteps=True)
                    else:
                        if not encoder_frozen:
                            freeze_encoder(model)
                            encoder_frozen = True
                        model.set_env(DummyVecEnv([env_fn]))
                        model.learn(_STEPS_PER_TASK, reset_num_timesteps=False)

                for j, (ename, eval_fn) in enumerate(_task_seq4):
                    eval_env = eval_fn()
                    mean_r, _ = evaluate_policy(
                        model, eval_env, n_eval_episodes=_N_EVAL_EPS, deterministic=False)
                    matrix[i, j] = float(mean_r)
                    eval_env.close()

            return matrix.tolist()

        exp4_results = {}
        for _cond4 in ("scratch", "plastic", "frozen"):
            print(f"\n=== Exp 4: condition = {_cond4} ===")
            exp4_results[_cond4] = _run_condition4(_cond4)

        exp4_meta = {"task_names": [n for n, _ in _task_seq4]}
        save_json(_results_path, exp4_results)
        save_json(_meta_path, exp4_meta)
        print("\nExp 4 complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 5: Is Updating the Representation Actually Useful?

    **Protocol**
    - Task A: no obstacles (baseline)
    - Task B: obstacle at (2,2) — encoder is updated jointly with policy
    - Task C: obstacle at (3,2) — three encoder conditions:
      - `frozen_from_A`: encoder frozen after Task A, never sees B
      - `updated_on_B`: encoder trained on A+B, then frozen for C
      - `scratch_C`: fresh model trained only on C

    **Question**: does the encoder updated on B make learning on C faster?
    """)
    return


@app.cell
def _(
    BASE_PPO,
    Monitor,
    PPO,
    RESULTS_DIR,
    compute_aulc,
    freeze_encoder,
    load_json,
    make_obstacle_env,
    make_ppo,
    os,
    save_json,
    train_with_curve,
    transfer_all_policy,
    transfer_encoder,
):
    _curves_path = f"{RESULTS_DIR}/exp5/curves.json"
    _aulc_path = f"{RESULTS_DIR}/exp5/aulc.json"
    if os.path.exists(_curves_path) and os.path.exists(_aulc_path):
        print("=== Exp 5: loading existing results ===")
        exp5_curves = load_json(_curves_path)
        exp5_aulc = load_json(_aulc_path)
    else:
        _STEPS_A      = 100_000
        _STEPS_B      = 100_000
        _STEPS_C      = 100_000
        _EVAL_FREQ    = 5_000
        _N_EVAL_EPS   = 10

        _fn_A = lambda: Monitor(make_obstacle_env([]))
        _fn_B = lambda: Monitor(make_obstacle_env([(2, 2)]))
        _fn_C = lambda: Monitor(make_obstacle_env([(3, 2)]))

        print("=== Exp 5 ===")

        # ── Train Task A (shared across conditions) ───────────────────────────
        _model_A_path = f"{RESULTS_DIR}/exp5/model_A.zip"
        if os.path.exists(_model_A_path):
            print("  Loading existing model A")
            exp5_model_A = PPO.load(_model_A_path, env=_fn_A())
        else:
            print("Training Task A...")
            _env_A = _fn_A()
            exp5_model_A = make_ppo(_env_A, BASE_PPO)
            exp5_model_A.learn(_STEPS_A)
            exp5_model_A.save(f"{RESULTS_DIR}/exp5/model_A")

        # ── Continue on Task B (plastic encoder) ──────────────────────────────
        _model_AB_path = f"{RESULTS_DIR}/exp5/model_AB.zip"
        if os.path.exists(_model_AB_path):
            print("  Loading existing model AB")
            exp5_model_AB = PPO.load(_model_AB_path, env=_fn_B())
        else:
            print("Training Task B (continuing from A)...")
            _env_B = _fn_B()
            exp5_model_AB = make_ppo(_env_B, BASE_PPO)
            transfer_all_policy(exp5_model_A, exp5_model_AB)
            exp5_model_AB.learn(_STEPS_B, reset_num_timesteps=False)
            exp5_model_AB.save(f"{RESULTS_DIR}/exp5/model_AB")

        exp5_curves = {}

        # ── Condition 1: frozen_from_A ────────────────────────────────────────
        print("Task C – frozen_from_A...")
        _m = make_ppo(_fn_C(), BASE_PPO)
        transfer_encoder(exp5_model_A, _m)
        freeze_encoder(_m)
        exp5_curves["frozen_from_A"] = train_with_curve(
            _m, _STEPS_C, _fn_C, _EVAL_FREQ, _N_EVAL_EPS)

        # ── Condition 2: updated_on_B ─────────────────────────────────────────
        print("Task C – updated_on_B...")
        _m = make_ppo(_fn_C(), BASE_PPO)
        transfer_encoder(exp5_model_AB, _m)
        freeze_encoder(_m)
        exp5_curves["updated_on_B"] = train_with_curve(
            _m, _STEPS_C, _fn_C, _EVAL_FREQ, _N_EVAL_EPS)

        # ── Condition 3: scratch_C ────────────────────────────────────────────
        print("Task C – scratch_C...")
        _m = make_ppo(_fn_C(), BASE_PPO)
        exp5_curves["scratch_C"] = train_with_curve(
            _m, _STEPS_C, _fn_C, _EVAL_FREQ, _N_EVAL_EPS)

        exp5_aulc = {cond: compute_aulc(curve) for cond, curve in exp5_curves.items()}
        print(f"AULC: {exp5_aulc}")

        save_json(_curves_path, exp5_curves)
        save_json(_aulc_path, exp5_aulc)
        print("Exp 5 complete.")
    return


if __name__ == "__main__":
    app.run()
