import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from collections import defaultdict
    import warnings
    warnings.filterwarnings("ignore")
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    import cv2

    import minigrid
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    from stable_baselines3.common.monitor import Monitor
    import os

    from minigrid.core.grid import Grid
    from minigrid.core.world_object import Goal
    from minigrid.minigrid_env import MiniGridEnv
    from minigrid.core.mission import MissionSpace

    return (
        BaseFeaturesExtractor,
        Goal,
        Grid,
        ImgObsWrapper,
        MiniGridEnv,
        MissionSpace,
        Monitor,
        PPO,
        RGBImgObsWrapper,
        cv2,
        evaluate_policy,
        gym,
        nn,
        np,
        os,
        plt,
        torch,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluating generalization capabilities
    """)
    return


@app.cell
def _(cv2, gym, np):
    class AffineTransform(gym.ObservationWrapper):
        """
        T3: continuous affine transform (rotation + optional scale/translate).
        APPROXIMATELY bijective: bilinear interpolation introduces sub-pixel
        rounding error ≈ 1/256 per pixel.
        K ≈ 96 bits, NCD ≈ 0.03, NC1.
        Boundary: reflection padding (BORDER_REFLECT).

        Only meaningful on full RGB renders (not 7×7 tile encodings).

        Parameters
        ----------
        angle       : float  rotation angle in degrees (CCW)
        scale       : float  isotropic scaling factor
        translate   : tuple[int, int]  (tx, ty) translation in pixels
        interp      : str    'bilinear' | 'nearest'
        """

        def __init__(
            self,
            env: gym.Env,
            angle: float = 15.0,
            scale: float = 1.0,
            translate: tuple[int, int] = (0, 0),
            interp: str = "bilinear",
        ):
            super().__init__(env)
            self.angle = angle
            self.scale = scale
            self.translate = translate
            self.interp = interp

        def observation(self, obs: np.ndarray) -> np.ndarray:

            h, w = obs.shape[:2]
            flags = cv2.INTER_LINEAR if self.interp == "bilinear" else cv2.INTER_NEAREST

            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), self.angle, self.scale)
            M[0, 2] += self.translate[0]
            M[1, 2] += self.translate[1]

            return cv2.warpAffine(
                obs, M, (w, h),
                flags=flags,
                borderMode=cv2.BORDER_REFLECT,
            )
    # ─────────────────────────────────────────────────────────────────────────────
    # T1 – Integer Cyclic Translation
    # ─────────────────────────────────────────────────────────────────────────────

    class CyclicTranslation(gym.ObservationWrapper):
        """
        T1: translate image by (shift_x, shift_y) pixels with wraparound (np.roll).
        CRITICAL: uses cyclic/wraparound (np.roll), NOT zero-padding.
        Zero-padding destroys boundary pixels → Blackwell-degrading.
        K ≈ 14 bits, NCD ≈ 0.005, NC1.

        Parameters
        ----------
        shift_x : int   pixels shifted along the width  axis (axis=1)
        shift_y : int   pixels shifted along the height axis (axis=0)
        """

        def __init__(self, env: gym.Env, shift_x: int = 0, shift_y: int = 0):
            super().__init__(env)
            self.shift_x = shift_x
            self.shift_y = shift_y

        def observation(self, obs: np.ndarray) -> np.ndarray:
            out = np.roll(obs, self.shift_y, axis=0)
            out = np.roll(out, self.shift_x, axis=1)
            return out.copy()


    # ─────────────────────────────────────────────────────────────────────────────
    # T2 – Global Color Shift
    # ─────────────────────────────────────────────────────────────────────────────

    class GlobalColorShift(gym.ObservationWrapper):
        """
        T2: add constant (c_R, c_G, c_B) to every pixel mod 256.
        Exact bijection: inverse = subtract same constants mod 256.
        K ≈ 24 bits, NCD ≈ 0.005, NC0.
        Batch-normalised networks absorb the shift at the first BN layer (δ_IB ≈ 0).

        Parameters
        ----------
        shift : tuple[int, int, int]   per-channel additive constants in [0, 255]
        """

        def __init__(self, env: gym.Env, shift: tuple[int, int, int] = (50, 50, 50)):
            super().__init__(env)
            self.shift = np.array(shift, dtype=np.int32)

        def observation(self, obs: np.ndarray) -> np.ndarray:
            return ((obs.astype(np.int32) + self.shift) % 256).astype(np.uint8)



    class SpriteTextureSwap(gym.ObservationWrapper):
        """
        T4: replace sprites using MiniGrid's rendering primitives and grid API.
        Controllable via boolean flags:
          swap_agent     – agent triangle  →  key shape  (rotated with direction)
          swap_obstacles – interior walls  →  grass tiles (shape + colour changed)
          swap_goal      – goal square     →  treasure chest
          swap_floor     – empty cells     →  cobblestone texture
        Border walls are never modified.
        NON-BIJECTION: original sprites are discarded → Blackwell-degrading.
        K ≈ 6200 bits, NCD 0.05–0.3.
        """

        def __init__(
            self,
            env: gym.Env,
            swap_agent: bool = False,
            swap_obstacles: bool = False,
            swap_goal: bool = False,
            swap_floor: bool = False,
        ):
            super().__init__(env)
            self.swap_agent = swap_agent
            self.swap_obstacles = swap_obstacles
            self.swap_goal = swap_goal
            self.swap_floor = swap_floor

        # ── sprite draw helpers ───────────────────────────────────────────────

        @staticmethod
        def _draw_key(tile: np.ndarray, agent_dir: int) -> None:
            """Draw a key sprite rotated to match agent facing direction.

            The base key points downward (ring at top, teeth at bottom).
            agent_dir:  0=right, 1=down, 2=left, 3=up  (MiniGrid convention).
            """
            import math
            from minigrid.core.constants import COLORS
            from minigrid.utils.rendering import (
                fill_coords, point_in_rect, point_in_circle, rotate_fn,
            )

            c = COLORS["red"]
            theta = {0: -math.pi / 2, 1: 0, 2: math.pi / 2, 3: math.pi}[agent_dir]

            def rot(fn):
                return rotate_fn(fn, 0.5, 0.5, theta)

            fill_coords(tile, rot(point_in_rect(0.50, 0.63, 0.31, 0.88)), c)          # shaft
            fill_coords(tile, rot(point_in_rect(0.38, 0.50, 0.59, 0.66)), c)          # tooth 1
            fill_coords(tile, rot(point_in_rect(0.38, 0.50, 0.81, 0.88)), c)          # tooth 2
            fill_coords(tile, rot(point_in_circle(cx=0.56, cy=0.28, r=0.190)), c)     # ring outer
            fill_coords(tile, rot(point_in_circle(cx=0.56, cy=0.28, r=0.064)), (0, 0, 0))  # ring hole

        @staticmethod
        def _draw_grass(tile: np.ndarray) -> None:
            """Draw a grass tile: brown earth base with green blade triangles."""
            from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_triangle

            fill_coords(tile, point_in_rect(0, 1, 0, 1), (100, 70, 40))
            blades = [
                ((0.05, 0.95), (0.22, 0.95), (0.13, 0.20), (0, 160, 0)),
                ((0.20, 0.95), (0.42, 0.95), (0.30, 0.10), (0, 190, 0)),
                ((0.38, 0.95), (0.58, 0.95), (0.50, 0.30), (0, 150, 0)),
                ((0.55, 0.95), (0.75, 0.95), (0.62, 0.05), (34, 200, 34)),
                ((0.72, 0.95), (0.92, 0.95), (0.82, 0.25), (0, 170, 0)),
            ]
            for a, b, tip, colour in blades:
                fill_coords(tile, point_in_triangle(a, b, tip), colour)

        @staticmethod
        def _draw_chest(tile: np.ndarray) -> None:
            """Draw a treasure chest sprite."""
            from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle

            # Chest body (dark brown box)
            fill_coords(tile, point_in_rect(0.10, 0.90, 0.35, 0.90), (110, 60, 20))
            # Lid (slightly lighter, arched top)
            fill_coords(tile, point_in_rect(0.10, 0.90, 0.25, 0.45), (140, 80, 30))
            # Lid arch top
            fill_coords(tile, point_in_circle(cx=0.50, cy=0.30, r=0.22), (140, 80, 30))
            # Metal band across front
            fill_coords(tile, point_in_rect(0.10, 0.90, 0.44, 0.52), (180, 160, 50))
            # Vertical band (clasp strip)
            fill_coords(tile, point_in_rect(0.42, 0.58, 0.25, 0.90), (180, 160, 50))
            # Lock / keyhole (gold circle + black hole)
            fill_coords(tile, point_in_circle(cx=0.50, cy=0.65, r=0.09), (220, 190, 60))
            fill_coords(tile, point_in_circle(cx=0.50, cy=0.65, r=0.04), (0, 0, 0))

        @staticmethod
        def _draw_sand(tile: np.ndarray) -> None:
            """Draw a sand floor texture: warm sandy base with grain speckles."""
            from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle

            # Sandy base
            fill_coords(tile, point_in_rect(0, 1, 0, 1), (210, 180, 120))
            # Darker grain patches
            dk = (185, 155, 95)
            fill_coords(tile, point_in_circle(0.20, 0.25, 0.08), dk)
            fill_coords(tile, point_in_circle(0.70, 0.15, 0.06), dk)
            fill_coords(tile, point_in_circle(0.45, 0.55, 0.07), dk)
            fill_coords(tile, point_in_circle(0.15, 0.75, 0.05), dk)
            fill_coords(tile, point_in_circle(0.80, 0.70, 0.07), dk)
            fill_coords(tile, point_in_circle(0.55, 0.85, 0.06), dk)
            # Light highlights
            hi = (230, 205, 150)
            fill_coords(tile, point_in_circle(0.35, 0.15, 0.05), hi)
            fill_coords(tile, point_in_circle(0.60, 0.40, 0.06), hi)
            fill_coords(tile, point_in_circle(0.25, 0.60, 0.04), hi)
            fill_coords(tile, point_in_circle(0.85, 0.45, 0.05), hi)

        # ── main observation transform ────────────────────────────────────────

        def observation(self, obs: np.ndarray) -> np.ndarray:
            from minigrid.core.world_object import Goal, Wall

            out = obs.copy()
            base = self.unwrapped
            gh, gw = base.grid.height, base.grid.width
            ih, iw = obs.shape[:2]
            ax, ay = base.agent_pos

            def _tile_array(gx: int, gy: int) -> tuple[np.ndarray, slice, slice]:
                x0 = int(round(gx * iw / gw));       x1 = min(int(round((gx + 1) * iw / gw)), iw)
                y0 = int(round(gy * ih / gh));       y1 = min(int(round((gy + 1) * ih / gh)), ih)
                tile = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
                return tile, slice(y0, y1), slice(x0, x1)

            # ── Grid objects ──────────────────────────────────────────────────
            for gy in range(1, gh - 1):
                for gx in range(1, gw - 1):
                    obj = base.grid.get(gx, gy)
                    if self.swap_obstacles and isinstance(obj, Wall):
                        tile, sy, sx = _tile_array(gx, gy)
                        self._draw_grass(tile)
                        out[sy, sx] = tile
                    elif self.swap_goal and isinstance(obj, Goal):
                        tile, sy, sx = _tile_array(gx, gy)
                        self._draw_chest(tile)
                        out[sy, sx] = tile
                    elif self.swap_floor and obj is None and (gx, gy) != (ax, ay):
                        tile, sy, sx = _tile_array(gx, gy)
                        self._draw_sand(tile)
                        out[sy, sx] = tile

            # ── Agent ─────────────────────────────────────────────────────────
            if self.swap_agent:
                tile, sy, sx = _tile_array(ax, ay)
                self._draw_key(tile, base.agent_dir)
                out[sy, sx] = tile

            return out

    return AffineTransform, CyclicTranslation, SpriteTextureSwap


@app.cell
def _(BaseFeaturesExtractor, gym, nn, torch):

    class MinigridFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
            super().__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(observations))

    return (MinigridFeaturesExtractor,)


@app.cell
def _(os):
    def list_model_paths(models_dir: str):
        model_files = []
        for fname in os.listdir(models_dir):
            if fname.endswith(".zip") and fname.startswith("ppo_"):
                model_files.append(os.path.join(models_dir, fname))
        model_files.sort()
        return model_files

    def pretty_model_name(model_path: str):
        base = os.path.basename(model_path)
        return os.path.splitext(base)[0].replace("ppo_", "")

    return


@app.cell
def _(ImgObsWrapper, RGBImgObsWrapper, gym):
    # Example base env constructor
    def make_base_env():
        env = gym.make("MiniGrid-Empty-5x5-v0")
        env= RGBImgObsWrapper(env)
        env=ImgObsWrapper(env)
        return env

    return (make_base_env,)


@app.cell
def _(np, plt):
    def results_to_matrix(results):
        """
        Converte una struttura come:

        results = [
            ("train_env_1", [("eval_env_1", m11), ("eval_env_2", m12)]),
            ("train_env_2", [("eval_env_1", m21), ("eval_env_2", m22)]),
        ]

        in:
        - train_names
        - eval_names
        - matrix shape (n_train, n_eval)
        """
        train_names = [train_name for train_name, _ in results]

        # assume stesso ordine di eval per ogni train env
        eval_names = [eval_name for eval_name, _ in results[0][1]]

        matrix = np.zeros((len(train_names), len(eval_names)), dtype=float)

        for i, (_, eval_list) in enumerate(results):
            eval_dict = dict(eval_list)
            for j, eval_name in enumerate(eval_names):
                matrix[i, j] = eval_dict[eval_name]

        return train_names, eval_names, matrix


    def plot_results_matrix(results, figsize=(8, 6), cmap="viridis", annotate=True):
        train_names, eval_names, matrix = results_to_matrix(results)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

        ax.set_xticks(np.arange(len(eval_names)))
        ax.set_yticks(np.arange(len(train_names)))
        ax.set_xticklabels(eval_names, rotation=45, ha="right")
        ax.set_yticklabels(train_names)

        ax.set_xlabel("Evaluation environment")
        ax.set_ylabel("Training environment")
        ax.set_title("Train vs Eval performance matrix")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean reward")

        if annotate:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center"
                    )

        plt.tight_layout()
        plt.show()

        return train_names, eval_names, matrix

    return (plot_results_matrix,)


@app.cell
def _(PPO, evaluate_policy, os, policy_kwargs):
    def cross_eval(save_dir, envs):
        os.makedirs(save_dir, exist_ok=True)

        results = []

        for train_name, train_env_fn in envs:
            print(f"training on env {train_name}:")
            train_env = train_env_fn()
            print(train_env)
            model = PPO(
                "CnnPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=1e-4,     # or even 5e-5
                n_epochs=4,             # down from default 10
                target_kl=0.02,         # stop oversized policy jumps
                clip_range=0.2,
                clip_range_vf=0.2,      # stabilize critic too
                max_grad_norm=0.5,
                seed=0,
                device="cpu",
            )
            model.learn(100_000)

            # salva il modello
            model_path = os.path.join(save_dir, f"ppo_{train_name}.zip")
            model.save(model_path)
            print(f"saved model to {model_path}")

            evaluation_results = []
            for eval_name, eval_env_fn in envs:
                eval_env = eval_env_fn()
                print(eval_env)
                mean_reward, std_reward = evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=50,
                    deterministic=False,
                )
                evaluation_results.append((eval_name, mean_reward))
                eval_env.close()

            results.append((train_name, evaluation_results))
        train_env.close()
        return results

    return (cross_eval,)


@app.cell
def _(MinigridFeaturesExtractor):

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    return (policy_kwargs,)


@app.cell
def _(
    AffineTransform,
    CyclicTranslation,
    Monitor,
    SpriteTextureSwap,
    make_base_env,
):
    # One factory per transformation
    env_fns = [
        ("baseline", lambda: Monitor(make_base_env())),
        ("rot_30", lambda: Monitor(AffineTransform(make_base_env(), angle=30))),
        ("trans_x", lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8))),
        ("trans_y", lambda: Monitor(CyclicTranslation(make_base_env(), shift_y=8))),
        ("trans_xy", lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8, shift_y=8))),
        ("swap_agent", lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_agent=True))),
        ("swap_goal", lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_goal=True))),
        ("swap_floor", lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_floor=True))),
    ]
    return (env_fns,)


@app.cell
def _(cross_eval, env_fns):
    save_dir = "/zero_shot_models/obs_only"
    results= cross_eval(save_dir, env_fns)
    return (results,)


@app.cell
def _(plot_results_matrix, results):
    train_names, eval_names, matrix = plot_results_matrix(results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dynamics transfer
    """)
    return


@app.cell
def _(
    Goal,
    Grid,
    ImgObsWrapper,
    MiniGridEnv,
    MissionSpace,
    Monitor,
    RGBImgObsWrapper,
    np,
):


    class EmptyGridCustomGoal(MiniGridEnv):
        """5x5 empty gridworld with a configurable goal position."""

        def __init__(self, goal_pos=(3, 3), **kwargs):
            self.goal_pos = goal_pos
            mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")
            super().__init__(
                mission_space=mission_space,
                grid_size=5,
                max_steps=4 * 5 * 5,
                **kwargs,
            )

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.put_obj(Goal(), *self.goal_pos)
            self.place_agent()

    def make_env_goal_top_left():
        """Goal at (1,1) — top-left corner."""
        env = EmptyGridCustomGoal(goal_pos=(1, 1))
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    def make_env_goal_top_right():
        """Goal at (3,1) — top-right corner."""
        env = EmptyGridCustomGoal(goal_pos=(3, 1))
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    def make_env_goal_bottom_left():
        """Goal at (1,3) — bottom-left corner."""
        env = EmptyGridCustomGoal(goal_pos=(1, 3))
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    dynamics_envs = [
        ("goal_top_left", lambda: Monitor(make_env_goal_top_left())),
        ("goal_top_right", lambda : Monitor(make_env_goal_top_right())),
        ("goal_bottom_left",lambda:  Monitor(make_env_goal_bottom_left())),
    ]
    return (dynamics_envs,)


@app.cell
def _(cross_eval, dynamics_envs):
    save_dir_dyn = "/zero_shot_models/dyn_only"

    results_dyn = cross_eval(save_dir_dyn,dynamics_envs )
    return (results_dyn,)


@app.cell
def _(plot_results_matrix, results_dyn):
    train_names_dyn, eval_names_dyn, matrix_dyn = plot_results_matrix(results_dyn)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluate parallel generalization
    """)
    return


@app.cell
def _(cv2, gym, np):
    class ChangeEverythingWrapper(gym.ObservationWrapper):
        """
        Meta-wrapper: on each step, randomly samples one transformation
        from the pool of implemented transforms and applies it with
        random parameters.

        Includes: cyclic translation, global color shift, affine transform,
        and sprite texture swaps (agent, obstacles, goal, floor).
        """

        def __init__(self, env: gym.Env):
            super().__init__(env)
            self._transforms = [
                self._cyclic_translation,
                self._global_color_shift,
                self._affine_transform,
                self._sprite_swap_agent,
                self._sprite_swap_obstacles,
                self._sprite_swap_goal,
                self._sprite_swap_floor,
            ]

        def _cyclic_translation(self, obs: np.ndarray) -> np.ndarray:
            h, w = obs.shape[:2]
            shift_x = np.random.randint(-w // 2, w // 2)
            shift_y = np.random.randint(-h // 2, h // 2)
            out = np.roll(obs, shift_y, axis=0)
            out = np.roll(out, shift_x, axis=1)
            return out

        def _global_color_shift(self, obs: np.ndarray) -> np.ndarray:
            shift = np.random.randint(0, 256, size=3).astype(np.int32)
            return ((obs.astype(np.int32) + shift) % 256).astype(np.uint8)

        def _affine_transform(self, obs: np.ndarray) -> np.ndarray:
            angle = np.random.uniform(-180, 180)
            scale = np.random.uniform(0.8, 1.2)
            h, w = obs.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            return cv2.warpAffine(
                obs, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        # ── sprite helpers (reused from SpriteTextureSwap) ─────────────────

        def _tile_array(self, obs, gx, gy):
            base = self.unwrapped
            gh, gw = base.grid.height, base.grid.width
            ih, iw = obs.shape[:2]
            x0 = int(round(gx * iw / gw));       x1 = min(int(round((gx + 1) * iw / gw)), iw)
            y0 = int(round(gy * ih / gh));       y1 = min(int(round((gy + 1) * ih / gh)), ih)
            tile = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
            return tile, slice(y0, y1), slice(x0, x1)

        @staticmethod
        def _draw_key(tile, agent_dir):
            import math
            from minigrid.core.constants import COLORS
            from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle, rotate_fn
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
            for a, b, tip, colour in [
                ((0.05, 0.95), (0.22, 0.95), (0.13, 0.20), (0, 160, 0)),
                ((0.20, 0.95), (0.42, 0.95), (0.30, 0.10), (0, 190, 0)),
                ((0.38, 0.95), (0.58, 0.95), (0.50, 0.30), (0, 150, 0)),
                ((0.55, 0.95), (0.75, 0.95), (0.62, 0.05), (34, 200, 34)),
                ((0.72, 0.95), (0.92, 0.95), (0.82, 0.25), (0, 170, 0)),
            ]:
                fill_coords(tile, point_in_triangle(a, b, tip), colour)

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
            dk = (185, 155, 95)
            for cx, cy, r in [(0.20,0.25,0.08),(0.70,0.15,0.06),(0.45,0.55,0.07),
                               (0.15,0.75,0.05),(0.80,0.70,0.07),(0.55,0.85,0.06)]:
                fill_coords(tile, point_in_circle(cx, cy, r), dk)
            hi = (230, 205, 150)
            for cx, cy, r in [(0.35,0.15,0.05),(0.60,0.40,0.06),(0.25,0.60,0.04),(0.85,0.45,0.05)]:
                fill_coords(tile, point_in_circle(cx, cy, r), hi)

        # ── sprite swap transforms ─────────────────────────────────────────

        def _sprite_swap_agent(self, obs: np.ndarray) -> np.ndarray:
            base = self.unwrapped
            out = obs.copy()
            ax, ay = base.agent_pos
            tile, sy, sx = self._tile_array(obs, ax, ay)
            self._draw_key(tile, base.agent_dir)
            out[sy, sx] = tile
            return out

        def _sprite_swap_obstacles(self, obs: np.ndarray) -> np.ndarray:
            from minigrid.core.world_object import Wall
            base = self.unwrapped
            out = obs.copy()
            gh, gw = base.grid.height, base.grid.width
            for gy in range(1, gh - 1):
                for gx in range(1, gw - 1):
                    if isinstance(base.grid.get(gx, gy), Wall):
                        tile, sy, sx = self._tile_array(obs, gx, gy)
                        self._draw_grass(tile)
                        out[sy, sx] = tile
            return out

        def _sprite_swap_goal(self, obs: np.ndarray) -> np.ndarray:
            from minigrid.core.world_object import Goal
            base = self.unwrapped
            out = obs.copy()
            gh, gw = base.grid.height, base.grid.width
            for gy in range(1, gh - 1):
                for gx in range(1, gw - 1):
                    if isinstance(base.grid.get(gx, gy), Goal):
                        tile, sy, sx = self._tile_array(obs, gx, gy)
                        self._draw_chest(tile)
                        out[sy, sx] = tile
            return out

        def _sprite_swap_floor(self, obs: np.ndarray) -> np.ndarray:
            base = self.unwrapped
            out = obs.copy()
            gh, gw = base.grid.height, base.grid.width
            ax, ay = base.agent_pos
            for gy in range(1, gh - 1):
                for gx in range(1, gw - 1):
                    if base.grid.get(gx, gy) is None and (gx, gy) != (ax, ay):
                        tile, sy, sx = self._tile_array(obs, gx, gy)
                        self._draw_sand(tile)
                        out[sy, sx] = tile
            return out

        def observation(self, obs: np.ndarray) -> np.ndarray:
            transform = self._transforms[np.random.randint(len(self._transforms))]
            return transform(obs)

    return (ChangeEverythingWrapper,)


@app.cell
def _(ChangeEverythingWrapper, ImgObsWrapper, RGBImgObsWrapper, gym):
    def make_everything_env():

        env = env = gym.make("MiniGrid-Empty-5x5-v0")
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        env= ChangeEverythingWrapper(env)
        return env

    return (make_everything_env,)


@app.cell
def _(Monitor, make_everything_env):
    train_env= Monitor(make_everything_env())
    return (train_env,)


@app.cell
def _(PPO, policy_kwargs, train_env):
    model = PPO(
                "CnnPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=1e-4,     # or even 5e-5
                n_epochs=4,             # down from default 10
                target_kl=0.02,         # stop oversized policy jumps
                clip_range=0.2,
                clip_range_vf=0.2,      # stabilize critic too
                max_grad_norm=0.5,
                seed=0,
                device="cpu",
            )
    model.learn(800_000)
    return (model,)


@app.cell
def _(env_fns, evaluate_policy, model):
    results_everything=[]
    for eval_name, eval_env_fn in env_fns:
        eval_env = eval_env_fn()
        print(eval_env)
        mean_reward, std_reward = evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=50,
                    deterministic=False,
                )
        results_everything.append((eval_name, mean_reward))
        eval_env.close()

    return (results_everything,)


@app.cell
def _(model):
    model.save("./change_everything")
    return


@app.cell
def _(results_everything):
    results_everything
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluate continual generalization
    """)
    return


@app.cell
def _(PPO, evaluate_policy, os, policy_kwargs):
    def continual_train(train_stages, eval_envs, save_prefix, steps_per_stage=100_000):
        """
        Continual learning: train a single PPO model sequentially through
        a list of environments, then evaluate the final model on all eval envs.

        Parameters
        ----------
        train_stages    : list of (name, env_factory)
        eval_envs       : list of (name, env_factory)
        save_prefix     : str   path prefix for saving the final checkpoint
        steps_per_stage : int   training steps per stage

        Returns
        -------
        results : list of (eval_name, mean_reward)
        """
        os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)

        model = None
        for i, (stage_name, env_fn) in enumerate(train_stages):
            train_env = env_fn()
            if model is None:
                model = PPO(
                   "CnnPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=1e-4,     # or even 5e-5
                n_epochs=4,             # down from default 10
                target_kl=0.02,         # stop oversized policy jumps
                clip_range=0.2,
                clip_range_vf=0.2,      # stabilize critic too
                max_grad_norm=0.5,
                seed=0,
                device="cpu",
                )
            else:
                model.set_env(train_env)
            print(f"[stage {i}] training on {stage_name} for {steps_per_stage} steps")
            model.learn(steps_per_stage, reset_num_timesteps=False)
            train_env.close()

        model.save(f"{save_prefix}_final.zip")
        print(f"saved final model to {save_prefix}_final.zip")

        results = []
        for eval_name, eval_env_fn in eval_envs:
            eval_env = eval_env_fn()
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=50, deterministic=False,
            )
            results.append((eval_name, mean_reward))
            print(f"  {eval_name}: {mean_reward:.2f} +/- {std_reward:.2f}")
            eval_env.close()

        return results

    return (continual_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exp 1 – base → shift_x → shift_y → test shift_xy + all previous
    """)
    return


@app.cell
def _(CyclicTranslation, Monitor, continual_train, make_base_env):
    cl_train_stages_1 = [
        ("baseline",  lambda: Monitor(make_base_env())),
        ("trans_x",   lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8))),
        ("trans_y",   lambda: Monitor(CyclicTranslation(make_base_env(), shift_y=8))),
    ]

    cl_eval_envs_1 = [
        ("baseline",  lambda: Monitor(make_base_env())),
        ("trans_x",   lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8))),
        ("trans_y",   lambda: Monitor(CyclicTranslation(make_base_env(), shift_y=8))),
        ("trans_xy",  lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8, shift_y=8))),
    ]

    cl_results_1 = continual_train(
        cl_train_stages_1,
        cl_eval_envs_1,
        save_prefix="continual_models/exp1_base_tx_ty",
    )
    return (cl_results_1,)


@app.cell
def _(cl_results_1, plt):
    _names = [n for n, _ in cl_results_1]
    _rewards = [r for _, r in cl_results_1]
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bars = ax1.bar(_names, _rewards)
    ax1.set_ylabel("Mean reward")
    ax1.set_title("Exp 1: base → shift_x → shift_y")
    ax1.axhline(0, color="grey", linewidth=0.5)
    for bar, v in zip(bars, _rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exp 2 – base → shift_xy → test shift_x, shift_y + all previous
    """)
    return


@app.cell
def _(CyclicTranslation, Monitor, continual_train, make_base_env):
    cl_train_stages_2 = [
        ("baseline",  lambda: Monitor(make_base_env())),
        ("trans_xy",  lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8, shift_y=8))),
    ]

    cl_eval_envs_2 = [
        ("baseline",  lambda: Monitor(make_base_env())),
        ("trans_xy",  lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8, shift_y=8))),
        ("trans_x",   lambda: Monitor(CyclicTranslation(make_base_env(), shift_x=8))),
        ("trans_y",   lambda: Monitor(CyclicTranslation(make_base_env(), shift_y=8))),
    ]

    cl_results_2 = continual_train(
        cl_train_stages_2,
        cl_eval_envs_2,
        save_prefix="continual_models/exp2_base_txy",
    )
    return (cl_results_2,)


@app.cell
def _(cl_results_2, plt):
    _names = [n for n, _ in cl_results_2]
    _rewards = [r for _, r in cl_results_2]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    _bars = ax2.bar(_names, _rewards)
    ax2.set_ylabel("Mean reward")
    ax2.set_title("Exp 2: base → shift_xy")
    ax2.axhline(0, color="grey", linewidth=0.5)
    for _bar, _v in zip(_bars, _rewards):
        ax2.text(_bar.get_x() + _bar.get_width() / 2, _v + 0.01, f"{_v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exp 3 – base → swap_agent → swap_goal → swap_obstacle → swap_floor → test swap_all + all previous
    """)
    return


@app.cell
def _(Monitor, SpriteTextureSwap, continual_train, make_base_env):
    cl_train_stages_3 = [
        ("baseline",      lambda: Monitor(make_base_env())),
        ("swap_agent",    lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_agent=True))),
        ("swap_goal",     lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_goal=True))),
        ("swap_obstacle", lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_obstacles=True))),
        ("swap_floor",    lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_floor=True))),
    ]

    cl_eval_envs_3 = [
        ("baseline",      lambda: Monitor(make_base_env())),
        ("swap_agent",    lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_agent=True))),
        ("swap_goal",     lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_goal=True))),
        ("swap_obstacle", lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_obstacles=True))),
        ("swap_floor",    lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_floor=True))),
        ("swap_all",      lambda: Monitor(SpriteTextureSwap(make_base_env(), swap_agent=True, swap_obstacles=True, swap_goal=True, swap_floor=True))),
    ]

    cl_results_3 = continual_train(
        cl_train_stages_3,
        cl_eval_envs_3,
        save_prefix="continual_models/exp3_base_swaps",
    )
    return (cl_results_3,)


@app.cell
def _(cl_results_3, plt):
    _names = [n for n, _ in cl_results_3]
    _rewards = [r for _, r in cl_results_3]
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    _bars = ax3.bar(_names, _rewards)
    ax3.set_ylabel("Mean reward")
    ax3.set_title("Exp 3: base → swap_agent → swap_goal → swap_obstacle → swap_floor")
    ax3.axhline(0, color="grey", linewidth=0.5)
    for _bar, _v in zip(_bars, _rewards):
        ax3.text(_bar.get_x() + _bar.get_width() / 2, _v + 0.01, f"{_v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
