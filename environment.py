"""Implements the game loop and handles the user's events."""

import os
import random
import numpy as np
import pygame as pg
import constants as const

from utils import message, distance

vec = pg.math.Vector2
n_snap = 0

# Manually places the window
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (50, 50)

ACTION_SPACE = 4

MAX_FRAME = 250

REWARD_CLOSE_FOOD = 1
REWARD_EAT = 10

PENALTY_WANDER = -1
PENALTY_COLLISION = -10
PENALTY_FAR_FOOD = 0

MOVES = {0: "right", 1: "left", 2: "down", 3: "up"}


class Block(pg.sprite.Sprite):
    def __init__(self, x, y, w, h, color):
        pg.sprite.Sprite.__init__(self)

        self.pos = vec(x, y)
        self.color = color
        self.image = pg.Surface((w, h))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)


class Game:
    def __init__(self, human=False, grid=False, infos=True) -> None:
        pg.init()
        self.human = human
        self.grid = grid
        self.infos = infos
        self.screen = pg.display.set_mode([const.PLAY_WIDTH, const.PLAY_HEIGHT])
        self.clock = pg.time.Clock()

        pg.display.set_caption(const.TITLE)
                
        self.agent = Block(0, 0, const.BLOCK_SIZE, const.BLOCK_SIZE, pg.Color("Blue"))
        self.food = Block(0, 0, const.BLOCK_SIZE, const.BLOCK_SIZE, pg.Color("Green"))

        self.running = True
        self.n_games = 0
        self.n_frames_threshold = 0
        self.score = 0
        self.highest_score = 0
        self.sum_scores = 0
        self.sum_rewards = 0
        self.mean_scores = [0]
        self.mean_rewards = [0]
        self.reward_episode = 0

        self.direction = None
        self.dangerous_locations = set()
        self.distance_food = distance(self.agent.pos, self.food.pos)
        
        self.state_space = len(self.get_state())
        self.action_space = ACTION_SPACE

    def random_coordinates(self):
        idx_x = random.randint(0, const.PLAY_WIDTH // const.BLOCK_SIZE - 1)
        idx_y = random.randint(0, const.PLAY_HEIGHT // const.BLOCK_SIZE - 1)

        x = const.BLOCK_SIZE * (idx_x + 0.5)
        y = const.BLOCK_SIZE * (idx_y + 0.5)
        
        return x, y
    
    def place_entity(self, entity, other):
        x, y = self.random_coordinates()
        entity.pos = vec(x, y)
        entity.rect.center = entity.pos
        
        if entity.rect.colliderect(other.rect):
            self.place_entity(entity, other)
        
    def reset(self) -> np.array:
        """Resets the game and return its corresponding state."""
        self.score = 0
        self.n_frames_threshold = 0
        self.reward_episode = 0
        self.dangerous_locations.clear()
        self.place_entity(self.agent, other=self.food)
        self.place_entity(self.food, other=self.agent)

        return self.get_state()

    def move(self, action) -> None:
        """
        Moves player according to the action chosen by the model.

        args:
            action (int, required): action chosen by the human/agent to move the player
        """
        self.direction = MOVES[action]
        if self.direction == "right":  # going right
            self.agent.pos.x += const.AGENT_X_SPEED
        elif self.direction == "left":  # going left
            self.agent.pos.x += -const.AGENT_X_SPEED
        elif self.direction == "up":  # going down
            self.agent.pos.y += -const.AGENT_Y_SPEED
        elif self.direction == "down":  # going up
            self.agent.pos.y += const.AGENT_Y_SPEED

        # Updating pos
        self.agent.rect.center = self.agent.pos

    def step(self, action):
        self.n_frames_threshold += 1

        self.events()
        self.move(action)

        reward, done = self.get_reward()
        state = self.get_state()

        return state, reward, done

    def get_state(self) -> np.array:
        r_player, r_food = self.agent.rect, self.food.rect
        state = [
                # current direction
                # self.direction == "right",
                # self.direction == "left",
                # self.direction == "down",
                # self.direction == "up",
                # food relative position
                r_player.right <= r_food.left,  # food is right
                r_player.left >= r_food.right,  # food is left
                r_player.bottom <= r_food.top,  # food is bottom
                r_player.top >= r_food.bottom,  # food is up
                # dangers
                self.wall_collision(offset=const.BLOCK_SIZE),
            ]
        
        return np.array(state, dtype=np.float32)

    def get_reward(self) -> tuple:
        done = False
        reward = 0

        # stops episode if the player does nothing but wonder around
        if self.n_frames_threshold > MAX_FRAME:
            return PENALTY_WANDER, True

        # checking for failure (wall or enemy collision)
        if self.wall_collision(offset=0):
            return PENALTY_COLLISION, True

        # checking if player is getting closer to food
        self.old_distance_food = self.distance_food
        self.distance_food = distance(self.agent.pos, self.food.pos)
        if self.distance_food < self.old_distance_food:
            reward = REWARD_CLOSE_FOOD

        # checking if eat:
        if self.food_collision():
            self.score += 1
            self.n_frames_threshold = 0
            self.place_entity(self.food, other=self.agent)
            reward = REWARD_EAT

        return reward, done

    def wall_collision(self, offset):
        r = self.agent.rect
        return (
            r.left - offset < 0
            or r.right + offset > const.PLAY_WIDTH
            or r.top - offset < 0
            or r.bottom + offset > const.PLAY_HEIGHT
        )
    
    def food_collision(self):
        return self.agent.rect.colliderect(self.food.rect)

    def events(self):
        for event in pg.event.get():
            if (
                event.type == pg.QUIT
                or event.type == pg.KEYDOWN
                and event.key == pg.K_q
            ):
                self.running = False

    def render(self):
        """TODO"""

        self.screen.fill(const.BACKGROUND_COLOR)
        self.draw_entities()

        if self.grid:
            self.draw_grid()

        if self.infos:
            self.draw_infos()

        pg.display.flip()
        self.clock.tick(const.FPS)

    def draw_entities(self):
        """TODO"""
        self.agent.draw(self.screen)
        self.food.draw(self.screen)

    def draw_grid(self):
        """TODO"""
        for i in range(1, const.PLAY_WIDTH // const.BLOCK_SIZE):
            # vertical lines
            p_v1 = const.INFO_WIDTH + const.BLOCK_SIZE * i, 0
            p_v2 = const.INFO_WIDTH + const.BLOCK_SIZE * i, const.PLAY_HEIGHT

            # horizontal lines
            p_h1 = 0, const.BLOCK_SIZE * i
            p_h2 = const.PLAY_WIDTH, const.BLOCK_SIZE * i

            pg.draw.line(self.screen, const.GRID_COLOR, p_v1, p_v2)
            pg.draw.line(self.screen, const.GRID_COLOR, p_h1, p_h2)

def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, numpy and random.

    Args:
        seed: random seed
    """

    try:
        import torch
    except ImportError:
        print("Module PyTorch cannot be imported")
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)


def main():
    pass


if __name__ == "__main__":
    main()
