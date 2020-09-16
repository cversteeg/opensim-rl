import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from .osim import OsimEnv

class Arm3DEnvMoBL(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/MoBL_ARMS_J_Simple_032118.osim')    
    time_limit = 200
    target_x = 0
    target_y = 0
    target_z = 0

    def get_observation(self):
        state_desc = self.get_state_desc()

        res = [self.target_x, self.target_y, self.target_z]

        # for body_part in ["r_humerus", "r_ulna_radius_hand"]:
        #     res += state_desc["body_pos"][body_part][0:2]
        #     res += state_desc["body_vel"][body_part][0:2]
        #     res += state_desc["body_acc"][body_part][0:2]
        #     res += state_desc["body_pos_rot"][body_part][2:]
        #     res += state_desc["body_vel_rot"][body_part][2:]
        #     res += state_desc["body_acc_rot"][body_part][2:]

        for joint in ["acromioclavicular", "elbow", "shoulder0", "shoulder1", "shoulder2", "sternoclavicular", "unrotscap"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in ["DELT1", "SUBSC", "TMAJ", "PECM2", "CORB", "TRIlong", "TRIlat", "BIClong", "BICshort", "BRA"]:
            res += [state_desc["muscles"][muscle]["activation"]]
            # res += [state_desc["muscles"][muscle]["fiber_length"]]
            # res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        res += state_desc["markers"]["Handle"]["pos"][:2]

        return res

    def get_observation_space_size(self):
        return 51 #46

    def generate_new_target(self):
        theta = random.uniform(0, math.pi)
        phi = random.uniform(-1*math.pi/2, math.pi/2)
        radius = random.uniform(0.3, 0.5)
        
        self.target_x = radius*math.sin(theta)*math.cos(phi)
        self.target_y = radius*math.sin(theta)*math.sin(phi) + 0.8
        self.target_z = radius*math.cos(theta)

        print('\nangeles: [{} {}]'.format(theta, phi))
        print('\ntarget: [{} {} {}]'.format(self.target_x, self.target_y, self.target_z))

        state = self.osim_model.get_state()

#        self.target_joint.getCoordinate(0).setValue(state, self.target_x, False)

        self.target_joint.getCoordinate(3).setLocked(state, False)
        self.target_joint.getCoordinate(3).setValue(state, self.target_x, False)
        self.target_joint.getCoordinate(3).setLocked(state, True)

        self.target_joint.getCoordinate(4).setLocked(state, False)
        self.target_joint.getCoordinate(4).setValue(state, self.target_y, False)
        self.target_joint.getCoordinate(4).setLocked(state, True)
        
        self.target_joint.getCoordinate(5).setLocked(state, False)
        self.target_joint.getCoordinate(5).setValue(state, self.target_z, False)
        self.target_joint.getCoordinate(5).setLocked(state, True)
        
      
        
        self.osim_model.set_state(state)
        
    def reset(self, random_target=True, obs_as_dict=True):
        obs = super(Arm3DEnvMoBL, self).reset(obs_as_dict=obs_as_dict)
        if random_target:
            self.generate_new_target()
        self.osim_model.reset_manager()
        return obs

    def __init__(self, *args, **kwargs):
        super(Arm3DEnvMoBL, self).__init__(*args, **kwargs)
        blockos = opensim.Body('target', 0.0001 , opensim.Vec3(0,0,0), opensim.Inertia(1,1,.0001,0,0,0) );
        self.target_joint = opensim.FreeJoint('target-joint',
                                  self.osim_model.model.getGround(), # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0),
                                  blockos, # PhysicalFrame
                                  opensim.Vec3(0, 0, -0.25),
                                  opensim.Vec3(0, 0, 0))

        self.noutput = self.osim_model.noutput

        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02);
        geometry.setColor(opensim.Green);
        blockos.attachGeometry(geometry)

        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)
        
        self.osim_model.model.initSystem()
    
    def reward(self):
        state_desc = self.get_state_desc()
        penalty = (state_desc["markers"]["Handle"]["pos"][0] - self.target_x)**2 + (state_desc["markers"]["Handle"]["pos"][1] - self.target_y)**2 +(state_desc["markers"]["Handle"]["pos"][2] - self.target_z)**2
        # vec2Targ = -1*[(state_desc["markers"]["Handle"]["pos"][0] - self.target_x), (state_desc["markers"]["Handle"]["pos"][1] - self.target_y)]
        # penaltyVel = (state_desc["markers"]["Handle"]["vel"][0],  state_desc["markers"]["Handle"]["vel"][1])
        act_pen = 0
        count = 0
        for muscle in ["DELT1", "SUBSC", "TMAJ", "PECM2", "CORB", "TRIlong", "TRIlat", "BIClong", "BICshort", "BRA"]:
            count +=1
            act_pen += state_desc["muscles"][muscle]["activation"]
            # res += [state_desc["muscles"][muscle]["fiber_length"]]
            # res += [state_desc["muscles"][muscle]["fiber_velocity"]]
            
        # print(state_desc["markers"]["r_radius_styloid"]["pos"])
        # print((self.target_x, self.target_y))
        act_pen = act_pen/count
        if np.isnan(penalty):
            penalty = 1
        return 1.-penalty

    def get_reward(self):
        return self.reward()


class Arm3DVecEnv(Arm3DEnvMoBL):
    def reset(self, obs_as_dict=False):
        obs = super(Arm3DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs
    def step(self, action, obs_as_dict=False):
        if np.isnan(action).any():
            action = np.nan_to_num(action)
        obs, reward, done, info = super(Arm3DVecEnv, self).step(.05*action, obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            done = True
            reward -10
        return obs, reward, done, info