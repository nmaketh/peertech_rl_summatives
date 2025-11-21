# environment/rendering.py

import pygame
from typing import Dict, Any, List, Tuple
import math
import random


class PeerTechRenderer:
    """
    Futuristic cartoon robot-girl renderer for the PeerTech RL environment.
    - 3D-style robot head
    - LED-style facial expressions (happy / neutral / sad)
    - Glow when mastery improves
    - Reward popups and confetti
    - Live stats panel on the right
    """

    def __init__(self, width: int = 1000, height: int = 650):
        pygame.init()
        pygame.display.set_caption("PeerTech – RL Robot Tutor")

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))

        # Fonts
        self.title_font = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.ui_font = pygame.font.SysFont("Segoe UI", 20)
        self.small_font = pygame.font.SysFont("Segoe UI", 16)

        self.clock = pygame.time.Clock()
        self.running = True

        # Expression / animation state
        self.expression_level = 2.0  # 0 = very sad, 2 = neutral, 4 = very happy
        self.target_expression = 2.0

        self.blink_timer = 0.0
        self.blink_interval = 2.5
        self.blink_duration = 0.08
        self.is_blinking = False

        self.head_tilt = 0.0
        self.head_tilt_vel = 0.0

        self.glow_timer = 0.0
        self.prev_mastery = None

        self.last_step_seen = -1

        # Visual effects
        self.reward_popups: List[Dict[str, Any]] = []
        self.confetti: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def reset(self):
        """Reset animation state between episodes."""
        self.expression_level = 2.0
        self.target_expression = 2.0
        self.blink_timer = 0.0
        self.is_blinking = False
        self.head_tilt = 0.0
        self.head_tilt_vel = 0.0
        self.glow_timer = 0.0
        self.prev_mastery = None
        self.last_step_seen = -1
        self.reward_popups.clear()
        self.confetti.clear()

    def render(self, state: Dict[str, Any]):
        """Main render entry (called from custom_env)."""
        if not self.running:
            return

        # Handle close button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        dt = self.clock.tick(60) / 1000.0

        # Update animation state
        self._update_animations(state, dt)

        # Draw everything
        self._draw_background()
        self._draw_avatar_panel(state)
        self._draw_stats_panel(state)
        self._draw_bottom_bar()
        self._draw_popups_and_confetti(dt)

        pygame.display.flip()

    def close(self):
        self.running = False
        pygame.quit()

    # =========================================================
    # Animation logic
    # =========================================================
    def _update_animations(self, state: Dict[str, Any], dt: float):
        reward = float(state.get("last_reward", 0.0))
        success = bool(state.get("last_success", False))
        step = int(state.get("step_count", 0))
        mastery = float(state.get("mastery", 0.0))
        engagement = float(state.get("engagement", 0.0))
        fatigue = float(state.get("fatigue", 0.0))

        # Expression target based on reward, engagement, fatigue
        base = 2.0  # neutral
        base += 1.5 * self._tanh_safe(reward / 4.0)
        base += 0.8 * (engagement - 0.5)
        base -= 0.8 * (fatigue - 0.3)
        self.target_expression = self._clamp(base, 0.0, 4.0)

        # Smooth movement towards target expression
        self.expression_level += (self.target_expression - self.expression_level) * 3.0 * dt

        # Blink timing
        self.blink_timer -= dt
        if not self.is_blinking and self.blink_timer <= 0.0:
            self.is_blinking = True
            self.blink_timer = self.blink_duration
        elif self.is_blinking and self.blink_timer <= 0.0:
            self.is_blinking = False
            self.blink_timer = self.blink_interval + random.uniform(-0.4, 0.4)

        # Head tilt (small shake on failure / big negative)
        if reward < -1.0 or (not success and reward < 0.0):
            self.head_tilt_vel = random.choice([-1.0, 1.0]) * 120.0

        self.head_tilt += self.head_tilt_vel * dt
        self.head_tilt_vel -= self.head_tilt * 6.0 * dt
        self.head_tilt *= (1.0 - 5.0 * dt)

        # Glow when mastery increases
        if self.prev_mastery is None:
            self.prev_mastery = mastery
        else:
            if mastery > self.prev_mastery + 0.01:
                self.glow_timer = 0.5
            self.prev_mastery = mastery

        if self.glow_timer > 0.0:
            self.glow_timer -= dt
            if self.glow_timer < 0.0:
                self.glow_timer = 0.0

        # One set of effects per env step
        if step != self.last_step_seen:
            self.last_step_seen = step
            if abs(reward) > 1e-3:
                self._spawn_reward_popup(reward)
                if reward > 2.5 and success:
                    self._spawn_confetti_burst()

    # =========================================================
    # Drawing layers
    # =========================================================
    def _draw_background(self):
        # Simple vertical gradient
        top_color = (20, 40, 90)
        bottom_color = (60, 20, 80)

        for y in range(self.height):
            t = y / self.height
            r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
            g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
            b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))

        # Top bar
        pygame.draw.rect(self.screen, (10, 15, 35), (0, 0, self.width, 60))
        title = self.title_font.render("PeerTech – RL Robot Tutor", True, (255, 255, 255))
        self.screen.blit(title, (20, 15))

    def _draw_avatar_panel(self, state: Dict[str, Any]):
        panel = pygame.Rect(40, 80, 460, 420)
        pygame.draw.rect(self.screen, (18, 24, 50), panel, border_radius=22)
        pygame.draw.rect(self.screen, (230, 230, 255), panel, 2, border_radius=22)

        label = self.ui_font.render("Robot Tutor Avatar", True, (235, 235, 255))
        self.screen.blit(label, (panel.x + 20, panel.y + 10))

        cx = panel.x + panel.w // 2
        cy = panel.y + panel.h // 2 + 20

        self._draw_robot_girl(cx, cy, state)

    def _draw_robot_girl(self, cx: int, cy: int, state: Dict[str, Any]):
        head_radius = 80
        surf_size = head_radius * 3
        surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        center = (surf_size // 2, surf_size // 2)

        mastery = float(state.get("mastery", 0.0))
        engagement = float(state.get("engagement", 0.0))
        fatigue = float(state.get("fatigue", 0.0))

        # Glow halo for mastery gain
        if self.glow_timer > 0.0:
            alpha = int(150 * self.glow_timer)
            pygame.draw.circle(
                surf,
                (120, 210, 255, alpha),
                center,
                head_radius + 30,
            )

        # Robot "helmet"
        helmet_color = (170, 200, 230)
        helmet_shadow = (120, 150, 190)
        pygame.draw.circle(surf, helmet_shadow, center, head_radius + 10)
        pygame.draw.circle(surf, helmet_color, center, head_radius)

        # Side "ears" / antenna bases
        pygame.draw.circle(surf, (90, 120, 170), (center[0] - head_radius - 8, center[1] - 10), 12)
        pygame.draw.circle(surf, (90, 120, 170), (center[0] + head_radius + 8, center[1] - 10), 12)

        # LED face panel (rectangle)
        face_width = int(head_radius * 1.6)
        face_height = int(head_radius * 1.1)
        face_rect = pygame.Rect(
            center[0] - face_width // 2,
            center[1] - face_height // 2,
            face_width,
            face_height,
        )
        pygame.draw.rect(surf, (15, 25, 45), face_rect, border_radius=18)

        # Eyes (LED strips or dots)
        eye_y = face_rect.y + face_height // 3
        eye_dx = face_width // 4

        if self.is_blinking:
            # Single LED strips for closed eyes
            pygame.draw.line(
                surf,
                (80, 200, 255),
                (center[0] - eye_dx - 12, eye_y),
                (center[0] - eye_dx + 12, eye_y),
                3,
            )
            pygame.draw.line(
                surf,
                (80, 200, 255),
                (center[0] + eye_dx - 12, eye_y),
                (center[0] + eye_dx + 12, eye_y),
                3,
            )
        else:
            # Small LED "eyes"
            pygame.draw.circle(surf, (120, 220, 255), (center[0] - eye_dx, eye_y), 7)
            pygame.draw.circle(surf, (120, 220, 255), (center[0] + eye_dx, eye_y), 7)
            pygame.draw.circle(surf, (230, 255, 255), (center[0] - eye_dx, eye_y - 2), 3)
            pygame.draw.circle(surf, (230, 255, 255), (center[0] + eye_dx, eye_y - 2), 3)

        # Eyebrow lights – show emotion
        express = self.expression_level - 2.0  # [-2, 2]
        brow_y = eye_y - 14
        brow_angle = express * 12.0
        self._draw_led_brow(surf, center[0] - eye_dx, brow_y, -brow_angle)
        self._draw_led_brow(surf, center[0] + eye_dx, brow_y, brow_angle)

        # LED mouth – curved line
        mouth_y = face_rect.y + int(face_height * 0.75)
        smile = self._clamp((self.expression_level - 2.0) / 2.0, -1.0, 1.0)
        points = []
        mouth_width = face_width * 0.6
        mouth_height = 25
        for i in range(12):
            t = i / 11.0
            x = center[0] - mouth_width / 2 + mouth_width * t
            # smile > 0 → curve up; smile < 0 → curve down
            y = mouth_y + (-smile) * (t - 0.5) ** 2 * mouth_height + (smile * 8)
            points.append((x, y))
        pygame.draw.lines(surf, (120, 220, 255), False, points, 3)

        # Chest / body
        body_rect = pygame.Rect(
            center[0] - 60,
            center[1] + head_radius - 10,
            120,
            70,
        )
        pygame.draw.rect(surf, (150, 190, 230), body_rect, border_radius=16)
        pygame.draw.rect(surf, (80, 120, 170), body_rect, 2, border_radius=16)

        # Central chest LED bar (energy)
        eng_ratio = self._clamp(engagement, 0.0, 1.0)
        fat_ratio = self._clamp(fatigue, 0.0, 1.0)
        bar_w = 80
        bar_h = 10
        bar_x = center[0] - bar_w // 2
        bar_y = body_rect.y + 20
        pygame.draw.rect(surf, (10, 20, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        # Engagement = blue from left
        pygame.draw.rect(
            surf,
            (90, 210, 255),
            (bar_x, bar_y, int(bar_w * eng_ratio), bar_h),
            border_radius=6,
        )
        # Fatigue = red from right
        pygame.draw.rect(
            surf,
            (255, 90, 110),
            (
                bar_x + bar_w - int(bar_w * fat_ratio),
                bar_y,
                int(bar_w * fat_ratio),
                bar_h,
            ),
            border_radius=6,
        )

        # Mastery stars under body
        mastery_stars = int(1 + mastery * 4)  # 1..5
        sx_start = center[0] - 40
        sy = body_rect.y + 45
        for i in range(mastery_stars):
            self._draw_star(surf, (sx_start + i * 20, sy), 6, (255, 220, 120))

        # Rotate whole robot head for tilt effect
        rotated = pygame.transform.rotate(surf, self.head_tilt)
        rect = rotated.get_rect(center=(cx, cy))
        self.screen.blit(rotated, rect)

    def _draw_led_brow(self, surf: pygame.Surface, x: int, y: int, angle_deg: float):
        length = 28
        rad = math.radians(angle_deg)
        dx = math.cos(rad) * length / 2
        dy = math.sin(rad) * length / 2
        start = (x - dx, y - dy)
        end = (x + dx, y + dy)
        pygame.draw.line(surf, (90, 210, 255), start, end, 3)

    def _draw_stats_panel(self, state: Dict[str, Any]):
        panel = pygame.Rect(520, 80, 430, 420)
        pygame.draw.rect(self.screen, (12, 26, 56), panel, border_radius=22)
        pygame.draw.rect(self.screen, (230, 230, 255), panel, 2, border_radius=22)

        label = self.ui_font.render("Live Learning Telemetry", True, (235, 235, 255))
        self.screen.blit(label, (panel.x + 20, panel.y + 10))

        mastery = float(state.get("mastery", 0.0))
        engagement = float(state.get("engagement", 0.0))
        fatigue = float(state.get("fatigue", 0.0))
        peer = float(state.get("peer_match", 0.0))
        success_rate = float(state.get("past_success_rate", 0.0))
        difficulty_norm = float(state.get("difficulty_norm", 0.0))
        subject_vec = state.get("subject_one_hot", [1.0, 0.0, 0.0])
        step = int(state.get("step_count", 0))
        last_action = state.get("last_action", None)
        last_reward = float(state.get("last_reward", 0.0))
        last_success = bool(state.get("last_success", False))

        subjects = ["Math", "Physics", "ICT"]
        try:
            subj_idx = int(max(range(3), key=lambda i: subject_vec[i]))
        except Exception:
            subj_idx = 0
        subject_label = subjects[subj_idx]

        bar_x = panel.x + 30
        bar_y = panel.y + 60
        bar_w = 250
        bar_h = 18
        gap = 40

        def draw_bar(y, val, label_text, color):
            v = self._clamp(val, 0.0, 1.0)
            pygame.draw.rect(self.screen, (230, 230, 255),
                             (bar_x, y, bar_w, bar_h), 2, border_radius=8)
            inner_w = int(bar_w * v)
            if inner_w > 0:
                pygame.draw.rect(self.screen, color,
                                 (bar_x + 2, y + 2, inner_w - 4, bar_h - 4),
                                 border_radius=8)
            text = self.small_font.render(f"{label_text}: {val:.2f}", True, (235, 235, 255))
            self.screen.blit(text, (bar_x, y - 20))

        draw_bar(bar_y + 0 * gap, mastery, "Mastery", (120, 220, 140))
        draw_bar(bar_y + 1 * gap, engagement, "Engagement", (110, 210, 255))
        draw_bar(bar_y + 2 * gap, fatigue, "Fatigue", (250, 120, 130))
        draw_bar(bar_y + 3 * gap, peer, "Peer Match", (255, 210, 120))
        draw_bar(bar_y + 4 * gap, success_rate, "Past Success", (180, 160, 255))

        # Difficulty + subject panel
        diff_level = 1 + int(round(self._clamp(difficulty_norm, 0.0, 1.0) * 4))
        diff_panel = pygame.Rect(panel.x + 310, panel.y + 70, 90, 180)
        pygame.draw.rect(self.screen, (7, 15, 35), diff_panel, border_radius=12)
        pygame.draw.rect(self.screen, (230, 230, 255), diff_panel, 2, border_radius=12)

        diff_text = self.small_font.render("Difficulty", True, (235, 235, 255))
        self.screen.blit(diff_text, (diff_panel.x + 6, diff_panel.y + 4))

        for i in range(diff_level):
            bx = diff_panel.x + 10
            by = diff_panel.y + 30 + (4 - i) * 18
            pygame.draw.rect(
                self.screen,
                (160, 120 + i * 20, 200 - i * 10),
                (bx, by, 70, 14),
                border_radius=4,
            )

        subj_label = self.small_font.render("Subject", True, (235, 235, 255))
        self.screen.blit(subj_label, (diff_panel.x + 6, diff_panel.y + 130))
        subj_val = self.small_font.render(subject_label, True, (255, 240, 200))
        self.screen.blit(subj_val, (diff_panel.x + 6, diff_panel.y + 148))

        # Episode / step info
        info_x = panel.x + 30
        info_y = panel.y + panel.h - 100

        reward_color = (130, 230, 140) if last_reward > 0 else \
            (250, 140, 140) if last_reward < 0 else (230, 230, 255)

        lines = [
            f"Step: {step}",
            f"Last action: {last_action}",
            f"Last reward: {last_reward:+.2f}",
            f"Outcome: {'Success' if last_success else 'Fail / Incomplete'}",
        ]
        for i, line in enumerate(lines):
            color = reward_color if "reward" in line else (230, 230, 255)
            txt = self.small_font.render(line, True, color)
            self.screen.blit(txt, (info_x, info_y + i * 20))

    def _draw_bottom_bar(self):
        bar = pygame.Rect(0, self.height - 70, self.width, 70)
        pygame.draw.rect(self.screen, (8, 12, 30), bar)

        tips = [
            "Controls:  ← easy   ↓ medium   → hard   SPACE: switch subject",
            "           A: high-compat peer   S: low-compat peer",
            "Mode: H = Human   J = AI   |   1=A2C  2=PPO  3=DQN  4=REINFORCE",
        ]
        for i, t in enumerate(tips):
            txt = self.small_font.render(t, True, (220, 220, 235))
            self.screen.blit(txt, (20, bar.y + 6 + i * 18))

    # =========================================================
    # Popups & Confetti
    # =========================================================
    def _spawn_reward_popup(self, reward: float):
        self.reward_popups.append({
            "x": self.width // 2,
            "y": 120,
            "text": f"{reward:+.2f}",
            "color": (90, 255, 140) if reward > 0 else (255, 120, 120),
            "life": 0.0,
            "max_life": 1.2,
        })

    def _spawn_confetti_burst(self):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(90, 190)
            self.confetti.append({
                "x": self.width // 2,
                "y": 140,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.uniform(0.4, 1.0),
                "max_life": random.uniform(0.8, 1.4),
                "color": random.choice([
                    (255, 220, 140),
                    (140, 210, 255),
                    (255, 150, 190),
                    (180, 255, 180),
                ]),
            })

    def _draw_popups_and_confetti(self, dt: float):
        # Reward popups
        for p in list(self.reward_popups):
            p["life"] += dt
            t = p["life"] / p["max_life"]
            if t >= 1.0:
                self.reward_popups.remove(p)
                continue
            p["y"] -= 40 * dt
            alpha = int(255 * (1.0 - t))

            text_surf = self.ui_font.render(p["text"], True, p["color"])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (p["x"] - text_surf.get_width() // 2, p["y"]))

        # Confetti particles
        for c in list(self.confetti):
            c["life"] -= dt
            if c["life"] <= 0.0:
                self.confetti.remove(c)
                continue

            c["x"] += c["vx"] * dt
            c["y"] += c["vy"] * dt
            c["vy"] += 200.0 * dt  # gravity

            radius = 3
            color = c["color"]          # (r, g, b)
            life_ratio = max(0.0, min(1.0, c["life"] / c["max_life"]))
            alpha = int(255 * life_ratio)

            particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, (color[0], color[1], color[2], alpha),
                               (radius, radius), radius)
            self.screen.blit(particle_surf, (c["x"], c["y"]))

    # =========================================================
    # Utilities
    # =========================================================
    def _draw_star(self, surf: pygame.Surface, center: Tuple[int, int],
                   radius: int, color: Tuple[int, int, int]):
        cx, cy = center
        points = []
        for i in range(5):
            angle = i * (2 * math.pi / 5) - math.pi / 2
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            points.append((x, y))
        pygame.draw.polygon(surf, color, points)

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _tanh_safe(self, x: float) -> float:
        return math.tanh(x)
