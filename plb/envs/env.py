from traceback import print_tb
import gym
from gym.spaces import Box
import os
import yaml
import numpy as np
from ..config import load
from yacs.config import CfgNode
from .utils import merge_lists
import taichi as ti

PATH = os.path.dirname(os.path.abspath(__file__))
def benchmark(func): 
    import time      
    def wrapper(*args, **kwargs):          
        t = time.perf_counter()          
        res = func(*args, **kwargs)          
        print(func.__name__, time.perf_counter()-t)          
        return res      
    return wrapper 

class PlasticineEnv(gym.Env):

    def __init__(self, cfg_path, version, nn=False):
        from ..engine.taichi_env import TaichiEnv
        self.cfg_path = cfg_path
        cfg = self.load_varaints(cfg_path, version)
        self.taichi_env = TaichiEnv(cfg, nn) # build taichi environment
        self.taichi_env.initialize()
        self.cfg = cfg.ENV
        self.taichi_env.set_copy(True)
        self._init_state = self.taichi_env.get_state()
        self._n_observed_particles = self.cfg.n_observed_particles

        obs = self.reset()
        self.observation_space = Box(-np.inf, np.inf, obs.shape)
        self.action_space = Box(-1, 1, (self.taichi_env.primitives.action_dim,))

    def reset(self):
        self.taichi_env.set_state(**self._init_state)
        self._recorded_actions = []
        return self._get_obs()
    #@benchmark
    def _get_obs(self, t=0):
        x = self.taichi_env.simulator.get_x(t)
        v = self.taichi_env.simulator.get_v(t)
        outs = []
        for i in self.taichi_env.primitives:
            outs.append(i.get_state(t))
            #print('outs.shape ',len(outs))
        s = np.concatenate(outs)
        step_size = len(x) // self._n_observed_particles
        #print(np.concatenate((x[::step_size], v[::step_size]), axis=-1).reshape(-1).shape)
        #print(s)
        #print(s.reshape(-1).shape)
        return np.concatenate((np.concatenate((x[::step_size], v[::step_size]), axis=-1).reshape(-1), s.reshape(-1)))
    @benchmark
    def step(self, action):
        self.taichi_env.step(action)
        loss_info = self.taichi_env.compute_loss()
        self._recorded_actions.append(action)
        obs = self._get_obs()
        r = loss_info['reward']
        if np.isnan(obs).any() or np.isnan(r):
            if np.isnan(r):
                print('nan in r')
            import pickle, datetime
            with open(f'{self.cfg_path}_nan_action_{str(datetime.datetime.now())}', 'wb') as f:
                pickle.dump(self._recorded_actions, f)
            raise Exception("NaN..")
        return obs, r, False, loss_info

    @benchmark
    def render(self, mode='human'):
        return self.taichi_env.render(mode)

    def myrender(self,mode='human',obs=[]):
        print(obs.shape)
        print(obs[0],' and ',obs[1])
        img = np.zeros((self.taichi_env.renderer.image_res[0], self.taichi_env.renderer.image_res[1], 3), dtype=np.float32)
        print(img.shape)
        return img
        fov = 0.23
        self.taichi_env.renderer.color_buffer.fill(0)
        #elf.image_res = cfg.image_res
        #return self.taichi_env.render(mode)
        for u, v in self.taichi_env.renderer.color_buffer:
            pos = self.taichi_env.renderer.camera_pos
            d = ti.Vector([
                (2 * fov * (u + ti.random(ti.f32)) / self.taichi_env.renderer.image_res[1] -
                fov * self.taichi_env.renderer.aspect_ratio - 1e-5),
                2 * fov * (v + ti.random(ti.f32)) / self.taichi_env.renderer.image_res[1] - fov - 1e-5, -1.0
            ])
            self.taichi_env.renderer.color_buffer[u, v] += d
        img = np.zeros((self.taichi_env.renderer.image_res[0], self.taichi_env.renderer.image_res[1], 3), dtype=np.float32)
        #self.copy(img, spp)
        return img[:, ::-1].transpose(1, 0, 2) # opencv format for render..
    @classmethod
    def load_varaints(self, cfg_path, version):
        assert version >= 1
        cfg_path = os.path.join(PATH, cfg_path)
        cfg = load(cfg_path)
        variants = cfg.VARIANTS[version - 1]

        new_cfg = CfgNode(new_allowed=True)
        new_cfg = new_cfg._load_cfg_from_yaml_str(yaml.safe_dump(variants))
        new_cfg.defrost()
        if 'PRIMITIVES' in new_cfg:
            new_cfg.PRIMITIVES = merge_lists(cfg.PRIMITIVES, new_cfg.PRIMITIVES)
        if 'SHAPES' in new_cfg:
            new_cfg.SHAPES = merge_lists(cfg.SHAPES, new_cfg.SHAPES)
        cfg.merge_from_other_cfg(new_cfg)

        cfg.defrost()
        # set target path id according to version
        name = list(cfg.ENV.loss.target_path)
        name[-5] = str(version)
        cfg.ENV.loss.target_path = os.path.join(PATH, '../', ''.join(name))
        cfg.VARIANTS = None
        cfg.freeze()

        return cfg
