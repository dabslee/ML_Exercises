import numpy as np
import gymnasium as gym

'''
We create a SkirmishEnv where the agent, a "rogue," is fighting against a "fighter" opponent.
The two take turns, with the fighter always going first:
- The fighter ATTACKs when the rogue is within 1 space, HEALs otherwise if the fighter has less than max HP, and otherwise MOVEs.
- The rogue has access to three actions: ATTACK, HEAL, or MOVE.
'''

# Generalized class for combatants for future extensions
class Combatant:
    def __init__(self, max_hp, speed, position):
        self.max_hp = max_hp
        self.current_hp = max_hp
        self.speed = speed
        self.position = position
    def take_hit(self):
        self.current_hp -= 1
    def heal(self):
        if self.current_hp < self.max_hp:
            self.current_hp += 0.5
    def dead(self):
        return self.current_hp <= 0

class SkirmishEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None, arena_size=30):
        # Initialize the arena and combatants
        self.arena_size = arena_size

        # Initialize observation and action spaces
        # observation_space = (rogue_hp, rogue_position, fighter_hp, fighter_position)
        self.observation_space = gym.spaces.Box(low=np.zeros(4), high=np.array([np.inf, arena_size, np.inf, arena_size]))
        self.action_space = gym.spaces.Discrete(3)
        self.action_names = ["ATTACK", "HEAL", "MOVE"]

        # Initialize render settings
        self.window_size = 512
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.array([self.rogue.current_hp, self.rogue.position, self.fighter.current_hp, self.fighter.position,])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rogue = Combatant(max_hp=5, speed=2, position=0)
        self.fighter = Combatant(max_hp=3, speed=1, position=np.random.uniform(0, self.arena_size))
        self.render()
        return self._get_obs(), {}
    
    def step(self, action):
        # rogue takes turn
        if action == self.action_names.index("ATTACK") and np.abs(self.rogue.position - self.fighter.position) <= 1:
            self.fighter.take_hit()
        elif action == self.action_names.index("HEAL"):
            self.rogue.heal()
        elif action == self.action_names.index("MOVE"):
            forward_pos = np.clip(self.rogue.position + self.rogue.speed, 0, self.arena_size)
            backward_pos = np.clip(self.rogue.position - self.rogue.speed, 0, self.arena_size)
            if np.abs(self.fighter.position - backward_pos) > np.abs(self.fighter.position - forward_pos):
                self.rogue.position = backward_pos
            else:
                self.rogue.position = forward_pos

        # fighter takes turn
        fighter_action = "DIES"
        if not self.fighter.dead():
            if np.abs(self.rogue.position - self.fighter.position) <= 1:
                fighter_action = "ATTACK"
                self.rogue.take_hit()
            elif self.fighter.current_hp < self.fighter.max_hp:
                fighter_action = "HEAL"
                self.fighter.heal()
            else:
                fighter_action = "MOVE"
                if np.abs(self.fighter.position - self.rogue.position) <= self.fighter.speed:
                    self.fighter.position = self.rogue.position
                else:
                    self.fighter.position += np.sign(self.rogue.position - self.fighter.position) * self.fighter.speed

        # check for termination and reward
        terminated = self.rogue.dead() or self.fighter.dead()
        reward = 10 if self.fighter.dead() else (0 if self.rogue.dead() else 0)

        # render and return
        self.render(action, fighter_action)
        return self._get_obs(), reward, terminated, False, {}
    
    def render(self, action=None, fighter_action=None):
        if self.render_mode == "human":
            if action is not None: print("Rogue took action %s. Fighter took action %s." % (self.action_names[action], fighter_action))
            print("Rogue has %.3f/%.3f hp and position %.3f." % (self.rogue.current_hp, self.rogue.max_hp, self.rogue.position))
            print("Fighter has %.3f/%.3f hp and position %.3f." % (self.fighter.current_hp, self.fighter.max_hp, self.fighter.position))
            print()

gym.envs.registration.register(
     id="Skirmish-v0",
     entry_point="skirmish_env:SkirmishEnv",
     max_episode_steps=100,
)