# environment/custom_env.py

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.rendering import PeerTechRenderer


class PeerTechEnv(gym.Env):
    """
    PeerTech: RL Environment for an Adaptive Robot Tutor.

    The agent controls a tutoring policy for a learner across 3 subjects
    (Math, Physics, ICT) by choosing difficulty, subject, and peer-matching.

    ----------------------------------------------------------
    Observation space: Box(9,)
    ----------------------------------------------------------
    Index | Name                 | Range   | Description
    ------+----------------------+---------+-------------------------------
      0   | mastery_level        | [0, 1]  | Overall mastery of current subject
      1   | engagement           | [0, 1]  | How focused / motivated the learner is
      2   | fatigue              | [0, 1]  | Cognitive tiredness
      3   | peer_match_quality   | [0, 1]  | Quality of peer/partner match
      4   | past_success_rate    | [0, 1]  | EMA of recent successes
      5   | difficulty_norm      | [0, 1]  | Normalised difficulty (from 1–5 scale)
      6   | subject_math         | {0, 1}  | One-hot for subject
      7   | subject_physics      | {0, 1}
      8   | subject_ict          | {0, 1}

    ----------------------------------------------------------
    Action space: Discrete(6)
    ----------------------------------------------------------
    0 -> Give an EASY challenge        (difficulty ~1)
    1 -> Give a MEDIUM challenge      (difficulty ~3)
    2 -> Give a HARD challenge        (difficulty ~5)
    3 -> Switch to another subject
    4 -> High-compat peer assignment (improves peer_match_quality)
    5 -> Low / random peer           (may lower peer_match_quality)

    Episode ends when:
      - fatigue is too high
      - engagement collapses
      - mastery is high enough
      - max_steps is reached (truncation)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # Rendering
        self.render_mode = render_mode
        self.renderer: Optional[PeerTechRenderer] = (
            PeerTechRenderer() if render_mode == "human" else None
        )

        # Discrete 6 actions as described above
        self.action_space = spaces.Discrete(6)

        # 9D observation as described in the docstring
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        # Episode config
        self.max_steps: int = 200

        # Core state variables
        self.mastery_level: float = 0.0
        self.engagement: float = 0.0
        self.fatigue: float = 0.0
        self.peer_match_quality: float = 0.0
        self.past_success_rate: float = 0.0
        self.difficulty_norm: float = 0.0
        self.subject_one_hot: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Book-keeping
        self.step_count: int = 0
        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0
        self.last_success: bool = False

        # Gymnasium RNG (set when reset is called)
        self.np_random: np.random.Generator | None = None

    # ------------------------------------------------------------------
    # GYM RESET
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to a fresh student episode.
        """
        super().reset(seed=seed)
        # self.np_random is now a np.random.Generator
        rng = self.np_random

        # Initial student profile: moderately capable, semi-engaged, low fatigue.
        self.mastery_level = float(rng.uniform(0.2, 0.5))
        self.engagement = float(rng.uniform(0.4, 0.8))
        self.fatigue = float(rng.uniform(0.0, 0.3))
        self.peer_match_quality = float(rng.uniform(0.4, 0.7))
        self.past_success_rate = float(rng.uniform(0.4, 0.7))

        # Start around difficulty 3/5
        self.difficulty_norm = self._difficulty_to_norm(3)

        # Random starting subject
        idx = int(rng.integers(0, 3))
        self.subject_one_hot = np.zeros(3, dtype=np.float32)
        self.subject_one_hot[idx] = 1.0

        # Book-keeping
        self.step_count = 0
        self.last_reward = 0.0
        self.last_action = None
        self.last_success = False

        if self.renderer:
            self.renderer.reset()

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    # ------------------------------------------------------------------
    # GYM STEP
    # ------------------------------------------------------------------
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply one tutoring decision from the agent and update the simulated learner.
        """

        # SB3 often passes actions as numpy arrays (e.g., array([2]))
        if isinstance(action, np.ndarray):
            # robust conversion for shape () or (1,)
            try:
                action = int(action.item())
            except Exception:
                action = int(action[0])
        else:
            action = int(action)

        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.step_count += 1
        self.last_action = action

        prev_mastery = self.mastery_level
        prev_engagement = self.engagement

        # ------------------------------
        # 1) APPLY ACTION EFFECTS
        # ------------------------------
        if action in (0, 1, 2):
            # Map difficulty choice to 1–5 scale then normalise
            difficulty_map = {0: 1, 1: 3, 2: 5}
            difficulty = difficulty_map[action]
            self.difficulty_norm = self._difficulty_to_norm(difficulty)

        elif action == 3:
            # Switch subject (might slightly change engagement)
            self._switch_subject()

        elif action == 4:
            # Improve peer match quality (good partner)
            self.peer_match_quality = float(
                np.clip(
                    self.peer_match_quality
                    + self.np_random.uniform(0.10, 0.30),
                    0.5,
                    1.0,
                )
            )

        elif action == 5:
            # Poor / random peer − can be slightly harmful
            self.peer_match_quality = float(
                np.clip(
                    self.peer_match_quality
                    + self.np_random.uniform(-0.20, 0.10),
                    0.1,
                    0.8,
                )
            )

        # ------------------------------
        # 2) CHALLENGE OUTCOME
        # ------------------------------
        success_prob = self._compute_success_prob()
        success = bool(self.np_random.random() < success_prob)
        self.last_success = success

        if success:
            self._apply_success_update()
            reward_success = 2.0
        else:
            self._apply_failure_update()
            reward_success = -1.0

        # ------------------------------
        # 3) REWARD SHAPING
        # ------------------------------
        delta_mastery = self.mastery_level - prev_mastery
        delta_engagement = self.engagement - prev_engagement

        reward = 0.0

        # Strong signal for learning progress
        reward += 10.0 * delta_mastery

        # Base success/failure reward
        reward += reward_success

        # Encourage keeping engagement up
        reward += 1.0 if delta_engagement > 0 else -1.0

        # Penalise over-fatigue and disengagement
        if self.fatigue > 0.7:
            reward -= 2.0
        if self.engagement < 0.3:
            reward -= 2.0

        # Extra bonus for success with good peer match
        if success and self.peer_match_quality > 0.7:
            reward += 1.0

        reward = float(np.clip(reward, -5.0, 5.0))
        self.last_reward = reward

        # ------------------------------
        # 4) TERMINATION / TRUNCATION
        # ------------------------------
        terminated = bool(
            (self.fatigue > 0.85)
            or (self.engagement < 0.2)
            or (self.mastery_level > 0.95)
        )

        truncated = bool(self.step_count >= self.max_steps)

        obs = self._get_obs()

        # Reason for termination (useful for analysis)
        termination_reason = None
        if terminated:
            if self.fatigue > 0.85:
                termination_reason = "fatigue"
            elif self.engagement < 0.2:
                termination_reason = "disengagement"
            else:
                termination_reason = "mastery_completed"

        info: Dict[str, Any] = {
            "success_prob": success_prob,
            "success": success,
            "difficulty_norm": self.difficulty_norm,
            "termination_reason": termination_reason,
        }

        # Rendering (Cartoon robot, HUD, etc.)
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(self._get_render_state())

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # GYM RENDER / CLOSE
    # ------------------------------------------------------------------
    def render(self):
        """
        Gymnasium-compatible render call.

        NOTE: For this env we render continuously from step() when
        render_mode == "human", so this is mostly a no-op.
        """
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(self._get_render_state())
        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------
    # INTERNAL HELPERS – DYNAMICS
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.mastery_level,
                self.engagement,
                self.fatigue,
                self.peer_match_quality,
                self.past_success_rate,
                self.difficulty_norm,
                self.subject_one_hot[0],
                self.subject_one_hot[1],
                self.subject_one_hot[2],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _difficulty_to_norm(difficulty: int) -> float:
        """
        Map discrete difficulty 1..5 → [0, 1].
        """
        difficulty = int(np.clip(difficulty, 1, 5))
        return float((difficulty - 1) / 4.0)

    def _switch_subject(self):
        """
        Rotate to a different subject. Often gives a small engagement bump
        (novelty effect), but not always.
        """
        current = int(np.argmax(self.subject_one_hot))
        # jump 1 or 2 steps forward mod 3
        jump = int(self.np_random.integers(1, 3))
        new = (current + jump) % 3

        self.subject_one_hot = np.zeros(3, dtype=np.float32)
        self.subject_one_hot[new] = 1.0

        # Slight engagement adjustment
        self.engagement = float(
            np.clip(
                self.engagement + self.np_random.uniform(-0.05, 0.10),
                0.0,
                1.0,
            )
        )

    def _compute_success_prob(self) -> float:
        """
        Probability that the learner succeeds on the current challenge.

        Intuition:
        - Higher engagement & peer_match help
        - High fatigue hurts
        - If difficulty is far above mastery, success is unlikely
        - If mastery is far above difficulty, success is also reduced
          (boredom → sloppy mistakes)
        """
        # Base from engagement & peer match
        base = 0.25 + 0.35 * self.engagement + 0.20 * self.peer_match_quality

        # Difficulty vs mastery
        gap = self.difficulty_norm - self.mastery_level  # positive → too hard
        if gap > 0.2:
            base -= 0.5 * gap  # too hard → fail more
        if self.mastery_level - self.difficulty_norm > 0.4:
            base -= 0.2  # too easy → boredom mistakes

        # Fatigue penalty
        base -= 0.35 * self.fatigue

        # Clamp to a safe range
        return float(np.clip(base, 0.05, 0.95))

    def _apply_success_update(self):
        """
        On success:
        - Increase mastery (more if difficulty is slightly above mastery)
        - Slight increase in engagement
        - Slight increase in fatigue
        - Update EMA of past_success_rate
        """
        gap = self.difficulty_norm - self.mastery_level
        # +gain if slightly challenging, smaller gain otherwise
        gain = 0.04 + 0.03 * np.clip(gap, -0.1, 0.3)
        gain = float(np.clip(gain, 0.01, 0.08))

        self.mastery_level = float(np.clip(self.mastery_level + gain, 0.0, 1.0))
        self.engagement = float(np.clip(self.engagement + 0.05, 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue + 0.04, 0.0, 1.0))

        # EMA towards 1.0
        self.past_success_rate = float(
            0.8 * self.past_success_rate + 0.2 * 1.0
        )

    def _apply_failure_update(self):
        """
        On failure:
        - If problem was much too hard, mastery may regress slightly
        - Engagement drops, especially if difficulty >> mastery
        - Fatigue always increases
        - past_success_rate pushed down
        """
        gap = self.difficulty_norm - self.mastery_level  # >0 → too hard

        dm = -0.03 if gap > 0.2 else 0.0
        self.mastery_level = float(np.clip(self.mastery_level + dm, 0.0, 1.0))

        # Larger engagement hit if it was clearly too hard
        base_drop = 0.04
        extra = 0.05 if gap > 0.2 else 0.0
        self.engagement = float(
            np.clip(self.engagement - (base_drop + extra), 0.0, 1.0)
        )

        self.fatigue = float(np.clip(self.fatigue + 0.06, 0.0, 1.0))

        # EMA towards 0.0
        self.past_success_rate = float(
            0.8 * self.past_success_rate + 0.2 * 0.0
        )

    # ------------------------------------------------------------------
    # RENDER STATE FOR PeerTechRenderer
    # ------------------------------------------------------------------
    def _get_render_state(self) -> Dict[str, Any]:
        """
        Adapter that converts internal state into what PeerTechRenderer expects.
        """
        return {
            "mastery": self.mastery_level,
            "engagement": self.engagement,
            "fatigue": self.fatigue,
            "peer_match": self.peer_match_quality,
            "past_success_rate": self.past_success_rate,
            "difficulty_norm": self.difficulty_norm,
            "subject_one_hot": self.subject_one_hot,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "last_reward": self.last_reward,
            "last_success": self.last_success,
        }
