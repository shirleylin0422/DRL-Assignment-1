import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
import math
from collections import deque
# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!


class SimpleTaxiEnv():
    
    def __init__(self, grid_size=5, fuel_limit=50):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = random.randint(5, 10)
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        
        self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None

       
        self.obstacles = set()  # No obstacles in simple version
        self.destination = None
    
    def _generate_stations_and_obstacles_and_taxi_pos(self, obstacle_ratio):
        def is_connected(start, targets):
            visited = set()
            queue = deque([start])

            while queue:
                current = queue.popleft()
                visited.add(current)
                x, y = current

                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    next_pos = (nx, ny)
                    if (
                        0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                        next_pos not in self.obstacles and
                        next_pos not in visited
                    ):
                        queue.append(next_pos)

            return all(t in visited for t in targets)

        self.stations = []
        while len(self.stations) < 4:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in self.stations:
                self.stations.append(pos)

        total_cells = self.grid_size * self.grid_size
        num_obstacles = math.ceil(total_cells * obstacle_ratio)

        while True:
            self.obstacles = set()
            while len(self.obstacles) < num_obstacles:
                pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
                if pos not in self.stations:
                    self.obstacles.add(pos)

            available_positions = [
                (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                if (x, y) not in self.stations and (x, y) not in self.obstacles
            ]

            self.taxi_pos = random.choice(available_positions)
            
            if is_connected(self.taxi_pos, self.stations):
                break 

    def reset(self, obstacle_ratio=0.2):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.grid_size = random.randint(5, 10)
        self._generate_stations_and_obstacles_and_taxi_pos(obstacle_ratio)
        
        
        self.passenger_loc = random.choice([pos for pos in self.stations])
        
        
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        reward -=10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -=10
                    
        reward -= 0.1  

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}

        

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state

    def get_state_by_obs(self, obs, target_pos, passenger_pos_view_by_taxi, destination_pos_view_by_taxi):
        '''
        obs = (
                [0] taxi_row, 
                [1] taxi_col, 
                [2] self.stations[0][0],
                [3] self.stations[0][1] ,
                [4] self.stations[1][0],
                [5] self.stations[1][1],
                [6] self.stations[2][0],
                [7] self.stations[2][1],
                [8] self.stations[3][0],
                [9] self.stations[3][1],
                [10] obstacle_north, 
                [11] obstacle_south, 
                [12] obstacle_east, 
                [13] obstacle_west, 
                [14] passenger_look, 
                [15] destination_look)
        '''
        state = (
            # (obs[0],obs[1]), # taxi pos
            obs[10], # obstacle
            obs[11],
            obs[12],
            obs[13],
            (target_pos[0]-obs[0],target_pos[1]-obs[1]), #target relative pos
            target_pos == passenger_pos_view_by_taxi, # if current target is passenger pos
            target_pos == destination_pos_view_by_taxi # if current target is destination pos
        )
        return state
    
    def draw_env_txt(self, taxi_pos,   action=None, step=None, fuel=None, episode=0, 
                     potential=0, reward=0, shaped_reward_detail=[], log_path="env_log.txt"):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        with open(log_path, "a", encoding="utf-8") as f:
            grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
            # Place obstacles
            for oy, ox in self.obstacles:
                grid[oy][ox] = 'x'

            for sy, sx in self.stations:
                if (sy, sx) != self.passenger_loc and (sy, sx) != self.destination:
                    grid[sy][sx] = 'S'

            # Place passenger
            py, px = self.passenger_loc
            if not self.passenger_picked_up:
                grid[py][px] = 'P'
            
            # Place destination
            dy, dx = self.destination
            grid[dy][dx] = '🔴'
        
            # Place taxi
            ty, tx = taxi_pos
            if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
                grid[ty][tx] = '🚖'

            f.write(f"\nEpisode: {episode}\n")
            f.write(f"Step: {step}\n")
            f.write(f"Taxi Position after last action: ({ty}, {tx})\n")
            f.write(f"Passenger picked?: {self.passenger_picked_up}\n")
            f.write(f"Dest Position: ({self.destination})\n")
            f.write(f"Fuel Left: {fuel}\n")
            f.write(f"Last Action: {self.get_action_name(action)}\n")
            f.write(f"Reward: {reward}\n")
            f.write(f"potential: {potential}\n")
            f.write(f"-----------------Shaped reward detail:---------------\n")
            for detail in shaped_reward_detail:
                f.write(f"{detail}\n")


            # 寫入地圖
            for row in grid:
                f.write(" ".join(row) + "\n")
            f.write("\n")

    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        # Place obstacles
        for oy, ox in self.obstacles:
            grid[oy][ox] = 'x'
        
        for sy, sx in self.stations:
            if (sy, sx) != self.passenger_loc and (sy, sx) != self.destination:
                grid[sy][sx] = 'S'

        # Place passenger
        py, px = self.passenger_loc
        if not self.passenger_picked_up:
            grid[py][px] = 'P'
        
        # Place destination
        dy, dx = self.destination
        grid[dy][dx] = '🔴'
    
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        print(f"Step: {step}\n")
        print(f"Taxi Position: ({tx}, {ty})\n")
        print(f"Passenger picked?: {self.passenger_picked_up}\n")
        print(f"Fuel Left: {fuel}\n")
        print(f"Last Action: {self.get_action_name(action)}\n")


        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")