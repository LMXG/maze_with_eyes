import gym
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import pandas as pd
import random
from PIL import Image
import cv2
import pickle
import os
import sys
import time

class MazeEnv(gym.Env):
    map = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,1,1],
    [1,0,1,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1]
    ], dtype = np.int)
    # 0:path 1:wall
    
    #托尔曼原图
    #[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
    #[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
    #[1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1],
    #[1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1],
    #[1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    #[1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1],
    #[1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1],
    #[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #[1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1],
    #[1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1],
    #[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #[1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1],
    #[1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1]

    #体现模型优势
    #[1,1,1,1,1,1,1,1,1,1],
    #[1,1,1,1,0,1,1,0,1,1],
    #[1,0,0,0,0,0,0,0,1,1],
    #[1,1,0,1,0,1,1,0,1,1],
    #[1,1,0,1,0,0,0,0,0,1],
    #[1,1,0,0,0,1,1,0,1,1],
    #[1,0,0,1,0,1,1,0,1,1],
    #[1,1,0,0,0,0,0,0,0,1],
    #[1,1,0,1,0,1,1,0,1,1],
    #[1,1,1,1,1,1,1,1,1,1]

    #复杂情况
    #[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #[1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1],
    #[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #[1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1],
    #[1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1],
    #[1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1],
    #[1,0,0,1,0,1,1,0,1,1,1,1,0,0,0,1,1,0,1,1],
    #[1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1],
    #[1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1],
    #[1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,1,1],
    #[1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,1,1],
    #[1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1],
    #[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #[1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1],
    #[1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1],
    #[1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1],
    #[1,0,0,1,0,1,1,0,1,1,1,1,0,0,0,1,1,0,1,1],
    #[1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1],
    #[1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1],
    #[1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,1,1],
    #[1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,1,1],
    #[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    unitsize = 50
    viewer = None

    state_space = [tuple(s) for s in np.argwhere(map == 0)]
    action_space = ['n', 'e', 's', 'w']

    
    state_trans = None
    agent_state = (0,0)
    agent_observation_state = dict()
    reward_list = [(9,3,100,100)]

    def gen_reward(self, reward_list):
        # reward_list (x,y,r,g) position:(x,y) reward:r gamma(loss):g
        ref = {'x' : 0, 'y' : 1, 'r' : 2, 'g' : 3}
        reward = np.zeros((len(self.map), len(self.map[0])))
        for reward_item in reward_list:
            for y in range(len(reward)):
                for x in range(len(reward[y])):
                    reward[x,y] += (1 - self.map[y,x]) * max((reward_item[ref['r']] - reward_item[ref['g']] * (abs(reward_item[ref['y']] - y)+abs(reward_item[ref['x']] - x))), 0)
        return reward    

    def gen_state_trans(self):
        state_trans = pd.DataFrame(
            data = None,
            index = self.state_space,
            columns = self.action_space
        )
        for state in self.state_space:
            for action in self.action_space:
                n_state = np.array(state)
                if   action == 'n':     n_state += np.array([ 0, 1])
                elif action == 'e':     n_state += np.array([ 1, 0])
                elif action == 's':     n_state += np.array([ 0,-1])
                elif action == 'w':     n_state += np.array([-1, 0])
                
                if  self.map[tuple(n_state)] == 0      \
                and 0 <= n_state[0] < len(self.map[0])  \
                and 0 <= n_state[1] < len(self.map):
                    state_trans.loc[state, action] = tuple(n_state)
                else:
                    state_trans.loc[state, action] = tuple(state)
        return state_trans

    def __init__(self):
        self.reward = self.gen_reward(self.reward_list)
        #print(self.reward)
        self.state_trans = self.gen_state_trans()
        self.width = len(self.map[0])
        self.height = len(self.map)
    
    def render(self, mode = 'rgb_array', close = False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        unit = self.unitsize
        wnd_width = self.width * unit
        wnd_height = self.height * unit

        if self.viewer is None:
            self.viewer = rendering.Viewer(wnd_width, wnd_height)
            self.render_square = []
            for x in range(self.width):
                lis = list()
                for y in range(self.height):
                    r = rendering.make_polygon(
                        v = [
                            [x * unit, y * unit],
                            [(x + 1) * unit, y * unit],
                            [(x + 1) * unit, (y + 1) * unit],
                            [x * unit, (y + 1) * unit],
                            [x * unit, y * unit]
                        ]
                    )
                    self.viewer.add_geom(r)
                    lis.append(r)
                self.render_square.append(lis.copy())
            
            # 画 Target
            for p in self.reward_list:
                point = p[0], p[1]
                self.render_square[point[0]][point[1]].set_color(1, 0, 0)
                
            # 创建网格
            for c in range(self.width):
                line = rendering.Line((0,c*unit),(wnd_width,c*unit))
                line.set_color(0,0,0)
                self.viewer.add_geom(line)
            for r in range(self.height):
                line = rendering.Line((r*unit, 0), (r*unit, wnd_height))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)
            
            self.robot = rendering.make_circle(20)
            self.robotrans = rendering.Transform()
            #self.robotrans.set_translation(self.state[0] * unit + unit / 2, self.state[1] * unit + unit / 2)
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)
            self.viewer.add_geom(self.robot)
            #time.sleep(100)
            return self.viewer.render(return_rgb_array = mode == 'rgb_array')
        
        if self.state is None:
            return None

        
        for point in self.state_space:
            x, y = point
            y = self.height - y - 1
            #print(x,y)
            if self.map[point] == 1:
                self.render_square[x][y].set_color(0, 0, 0)
            else:
                if not (x, self.height - y - 1) in self.agent_observation_state.keys():
                    color = 0.5
                else:
                    mem_strength = len(self.agent_observation_state[(x,self.height - y - 1)])
                    color = 0.5 + 0.1 * mem_strength
                self.render_square[x][y].set_color(color, color, color)
            for rpoint in self.reward_list:
                x = rpoint[0]
                y = self.height - rpoint[1] - 1
                self.render_square[x][y].set_color(0, 1, 0)


        # 更新 Agent
        self.robotrans.set_translation(self.agent_state[0] * unit + unit / 2, (self.height - self.agent_state[1] - 1) * unit + unit / 2)
        
        return self.viewer.render(return_rgb_array= mode == 'rgb_array')
    
    

def rgb_array_to_image(rgb_array, cnt = 0, datapath = "."):
    img = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
    cv2.imwrite("%s/%06d.jpg" % (datapath, cnt), img)

class Agent:
    curiosity_level = 10            # initial curiosity level
    observation_space = dict()      # {state_space : [times]}
    action_space = []
    memory_times = 5
    REWARD_NOTMOVE = -100
    time = 0

    curiosity_L1 = 0.1
    curiosity_L2 = 0.1

    state = (1, 11)                 # (x,y)
    def __init__(self, Env):
        self.Env = Env
        self.Env.state = self.state
        self.action_space = Env.action_space
        self.state_trans = pd.DataFrame(       
            data = None,
            index = Env.state_space,       
            columns = Env.action_space     
        )
        for state in Env.state_space:
            for action in Env.action_space:
                n_state = np.array(state)
                if   action == 'n':     n_state += np.array([ 0, 1])
                elif action == 'e':     n_state += np.array([ 1, 0])
                elif action == 's':     n_state += np.array([ 0,-1])
                elif action == 'w':     n_state += np.array([-1, 0])
                self.state_trans.loc[state, action] = tuple(n_state)
        self.mem_reword_e = np.zeros((self.Env.height, self.Env.width), dtype = np.float)
        self.observation_space[self.state] = [0]
        self.observation_space_ = dict()
        self.Env.agent_state = self.state
    
    def reward(self, state, action):
        ret = -1
        n_state = self.state_trans[action][state]
        if n_state in self.observation_space.keys():
            ret += self.curiosity_level * (self.memory_times - len(self.observation_space[n_state]))
        else:
            ret += self.curiosity_level * self.memory_times
        if n_state == state:
            ret += self.REWARD_NOTMOVE
        ret += self.mem_reword_e[state]
        return ret
    
    def transform(self, state, action):
        #print(self.state_trans[action][state])
        n_state = self.state_trans[action][state]
        reward = self.reward(state, action)
        is_new = False if n_state in self.observation_space else True
        return n_state, reward, False, is_new
    
    def step(self, action):
        state = self.state
        next_state, reward, is_terminal, is_new = self.transform(state, action)
        next_state = self.Env.state_trans[action][state]
        self.state_trans[action][state] = next_state
        self.mem_reword_e[state] = self.Env.reward[state]
        self.state = next_state
        if is_new:
            self.curiosity_level -= self.curiosity_L1
        else:
            self.curiosity_level -= self.curiosity_L1 * (self.memory_times - len(self.observation_space[next_state]))
            self.curiosity_level += self.curiosity_L2 * len(self.observation_space[next_state])
        self.curiosity_level = max(0, self.curiosity_level)
        self.time += 1
        self.Env.agent_state = self.state
        self.remember(self.state)
        #print(self.observation_space[self.state])
        
        #rgb_array_to_image(self.Env.render(), 0, '.')
        #print(self.mem_reword_e)
        return next_state, reward, is_terminal, is_new
    


    def remember(self, state):
        if not state in self.observation_space.keys():
            self.observation_space[state] = []
        self.observation_space[state].append(self.time)
        if len(self.observation_space[state]) > self.memory_times:
            self.observation_space[state] = self.observation_space[state][1:]
        self.mem_reword_e[state] = self.Env.reward[state]
        self.Env.agent_observation_state = self.observation_space
    
    def value_iterate(self):
            # state_space = []
        # x0, y0 = env.state
        # for pos in env.observed_space:
        #     x, y = pos
        #     if abs(x0 - x) < 10 and abs(y0 - y) < 10:
        #         state_space.append(pos)
        state_space = self.observation_space.keys()
        #print(self.observation_space)
        action_space = self.action_space
        v_s = pd.Series(
            data=np.zeros(shape=len(state_space)),
            index=state_space
        )
        policy = pd.Series(index=state_space)
        gamma = 0.2
        times = 0
        while True:
            if times >= 3:
                break
            times += 1
            v_s_ = v_s.copy()
            for state in state_space:
                #if state == env.state:
                #    print("!!!")
                v_s_a = pd.Series()
                for action in action_space:
                    state_, reward, is_done, is_new = self.transform(state,action)
                    if not state_ in self.observation_space.keys() or state == state_:
                        v_s_a[action] = reward
                    else:
                        v_s_a[action] = reward + gamma*v_s_[state_]
                v_s[state] = v_s_a.sum()
                best_choice = v_s_a[v_s_a == v_s_a.max()].index
                policy[state] = best_choice
            if (np.abs(v_s_ - v_s) < 1e-6).all():
                print(v_s)
                break
        return policy

       

    def go(self):
        for action in self.action_space:
            self.state_trans[action][self.state] = self.Env.state_trans[action][self.state]
        # reiter = False
        # for i in self.observation_space.keys():
        #     if not i in self.observation_space_.keys() or len(self.observation_space_[i]) != len(self.observation_space[i]):
        #         reiter = True
        #         break
        reiter = True
        if reiter:
            self.policy = self.value_iterate()
        self.observation_space_ = self.observation_space.copy()
        #print(self.policy)
        print(self.curiosity_level)
        choose = np.random.choice(self.policy[self.state])
        self.step(choose)
        #print(choose)
        rgb_array_to_image(self.Env.render(), self.time, '.')




maze = MazeEnv()
agent = Agent(maze)
#while True:
while True:
    agent.go()
    agent.Env.render()
    if agent.time == None:
        agent.Env.map = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,1,0,1,0,0,0,1,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,1,1],
    [1,0,1,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1]
    ], dtype = np.int)
        agent.Env.state_trans = agent.Env.gen_state_trans()
        #agent.state = (1, 11)