import gymnasium as gym
import sys
import overcooked_ai_py
from PIL import Image
#from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_env import Overcooked, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.agents.agent import AgentPair, FixedPlanAgent, RandomAgent, SampleAgent
import cv2
def main():
    #print(gym.pprint_registry())
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(mdp, horizon=100)
    env = gym.make("Overcooked-v0",base_env = base_env, featurize_fn =base_env.featurize_state_mdp)
    ra = RandomAgent(all_actions=True)
    print(f'env_action_space {env.action_space}')  # Discrete(6)
    print(f'env state space {env.observation_space} ')  # Box(0, inf, (96, ))  # flattened obs

    env.unwrapped.reset()
    frames = []
    for i in range(50):
        a = env.action_space.sample()
        print(f'action {a}')
        img = env.unwrapped.render()
        env.unwrapped.step((a, a))
        frames.append(Image.fromarray(img))
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    output_path = "output.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],  # Add the rest of the frames
        duration=100,             # Duration between frames in milliseconds
        loop=0                    # 0 means infinite loop
    )
    return

if __name__ == '__main__':
    main()