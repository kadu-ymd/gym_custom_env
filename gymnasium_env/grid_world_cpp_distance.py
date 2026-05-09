from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

class GridWorldCPPDistanceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200, fov_size: int = 5):
        self.size = size
        self.fov_size = fov_size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps

        # Track visited cells
        self.visited = set()
        self.all_free_cells_array = None

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros((self.fov_size, self.fov_size), dtype=int)

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((self.fov_size, self.fov_size), dtype=np.float32),
                high=np.full((self.fov_size, self.fov_size), 2.0, dtype=np.float32),
                dtype=np.float32
            ),
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _get_obs(self):
        _, dx_norm, dy_norm = self._get_nearest_unvisited_info()
        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
                dx_norm,
                dy_norm
            ], dtype=np.float32),
            "neighbors": self._neighbors.astype(np.float32),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def set_neighbors(self, obstacles_locations):
        matrix = np.zeros((self.fov_size, self.fov_size), dtype=int)
        offset = self.fov_size // 2
        for i in range(self.fov_size):
            for j in range(self.fov_size):
                nx = self._agent_location[0] + (j - offset)
                ny = self._agent_location[1] + (i - offset)
                neighbor = np.array([nx, ny])
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    matrix[i][j] = 1
                elif any(np.array_equal(neighbor, loc) for loc in obstacles_locations):
                    matrix[i][j] = 1
                elif (nx, ny) in self.visited:
                    matrix[i][j] = 2
        self._neighbors = matrix

    def _get_nearest_unvisited_info(self):
        if self.all_free_cells_array is None:
            return 0.0, 0.0, 0.0
            
        unvisited = [cell for cell in self.all_free_cells_array if tuple(cell) not in self.visited]
        if not unvisited:
            return 0.0, 0.0, 0.0
            
        unvisited_arr = np.array(unvisited)
        vectors = unvisited_arr - self._agent_location
        distances = np.linalg.norm(vectors, axis=1)
        
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        dx, dy = vectors[min_idx]
        dx_norm = dx / self.size
        dy_norm = dy / self.size
        
        return min_dist, dx_norm, dy_norm

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self.visited = set()

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while (np.array_equal(obstacle_location, self._agent_location) or
                   any(np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations)):
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)

        # Map all free cells once per episode to optimize distance calculation
        all_cells = np.array([[x, y] for x in range(self.size) for y in range(self.size)])
        free_cells = []
        for cell in all_cells:
            if not any(np.array_equal(cell, loc) for loc in self.obstacles_locations):
                free_cells.append(cell)
        self.all_free_cells_array = np.array(free_cells)

        self.visited.add(tuple(self._agent_location))
        self.set_neighbors(self.obstacles_locations)
        
        # Initialize distance tracking
        self.previous_min_distance, _, _ = self._get_nearest_unvisited_info()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if any(np.array_equal(self._agent_location, loc) for loc in self.obstacles_locations):
            self._agent_location = old_location

        self.set_neighbors(self.obstacles_locations)
        self.count_steps += 1

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        # --- Distance Shaped Reward Function ---
        reward = 0.0

        if stayed_in_place:
            reward -= 0.2

        current_min_distance, _, _ = self._get_nearest_unvisited_info()

        # Potential-based distance shaping: +0.1 for moving closer, 0.0 for moving further/parallel
        if current_min_distance < self.previous_min_distance:
            reward += 0.1
        else:
            reward += 0.0 # Removed the backward penalty to prevent the agent from getting psychologically trapped in local minima
            
        self.previous_min_distance = current_min_distance

        if is_new_cell:
            coverage_bonus = 5.0 * self.coverage_ratio 
            reward += 1.0 + coverage_bonus
            self.visited.add(current_pos)
        else:
            # Revisit penalty is removed because distance shaping inherently penalizes 
            # moving away from unvisited cells. 
            pass

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage

        if full_coverage:
            reward += 100.0

        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            remaining_ratio = 1.0 - self.coverage_ratio
            reward -= (50.0 * remaining_ratio)
        else:
            truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw visited cells in light green
        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),  # light green
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw obstacles in black
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent as blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw coverage info text
        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True, (0, 0, 0)
        )
        canvas.blit(coverage_text, (5, 5))

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
