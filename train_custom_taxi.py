import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict
import random
from simple_custom_taxi_env import SimpleTaxiEnv

def tabular_q_learning_taxi(episodes=10000, alpha=0.3, gamma=0.8,
                              epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.999, fuel_limit=5000,
                              q_table=None, debug=False):


    env = SimpleTaxiEnv(fuel_limit=fuel_limit)
    action_size = 6

    if q_table is None:
        q_table = {}

    rewards_per_episode = []
    epsilon = epsilon_start

    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def potential_function(agent_pos, target_pos):
        # Manhattan distance based potential function
        dist = manhattan_distance(agent_pos, target_pos)
        return 1 - (dist / (env.grid_size*2 - 2))
    
    def potential_shaping(agent_pos, next_agent_pos, target_pos):
        next_potential = potential_function(next_agent_pos, target_pos)
        prev_potential = potential_function(agent_pos, target_pos)
        potential_shaping_value = gamma * next_potential - prev_potential
        return potential_shaping_value

    def set_obstacle_ratio(episode, episodes):
        ratio = episode / episodes

        if ratio < 0.1:
            return 0.0
        elif ratio < 0.3:
            return 0.1
        else:
            return 0.2
        
    def sort_stations_by_distance(cur_taxi_pos, stations):

        sorted_stations = sorted(stations, key=lambda s: manhattan_distance(cur_taxi_pos, s))
        return sorted_stations

        
    if debug:
        with open('env_log.txt', "w", encoding="utf-8") as f:
            f.write(f"")

    for episode in range(episodes):
        done = False
        total_reward = 0
        episode_step = 0
        station_visited = set()
        obstacle_ratio = set_obstacle_ratio(episode, episodes)
        obs, _ = env.reset(obstacle_ratio=obstacle_ratio)
        stations = [(obs[2],obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        sorted_stations = sort_stations_by_distance((obs[0],obs[1]), stations)
        target_station = sorted_stations[0]
        next_target_station = None
        passenger_pos_view_by_taxi = None
        destination_pos_view_by_taxi = None
        pickup_time = 0
        state =  env.get_state_by_obs(obs, target_station, passenger_pos_view_by_taxi, destination_pos_view_by_taxi)


        while not done:
            '''
            state = (
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
            # record current state
            cur_passenger_picked_up = env.passenger_picked_up
            cur_taxi_pos = (obs[0], obs[1])
            cur_obstable_north = obs[10]
            cur_obstable_south = obs[11]
            cur_obstable_east = obs[12]
            cur_obstable_west = obs[13]
            cur_passenger_look = obs[14]
            cur_destination_look = obs[15]
            
            if state not in q_table:
                q_table[state] = np.zeros(action_size)

            if random.uniform(0, 1)<epsilon:
                action = random.choice(range(6))
            else:
                action = np.argmax(q_table[state])

            next_obs, reward, done, _ = env.step(action)

            # if env.current_fuel <= 0: # truncated?
            #     break

            # record next state 
            next_passenger_picked_up = env.passenger_picked_up
            next_taxi_pos = (next_obs[0], next_obs[1])
            next_obstable_north = next_obs[10]
            next_obstable_south = next_obs[11]
            next_obstable_east = next_obs[12]
            next_obstable_west = next_obs[13]
            next_passenger_look = next_obs[14]
            next_destination_look = next_obs[15]

            actions = [
                "Move South", 
                "Move North", 
                "Move East", 
                "Move West", 
                "Pick Up", 
                "Drop Off"
            ]

            # TODO: Reward shaping
            ''' Think
            - 還沒接到乘客前 ，鼓勵taxi往四個station (從最近的點開始)，
                並且記錄嘗試過的點，
                接近未visited過的點且passenger_look，鼓勵
                重複接近visited過且沒偵測到passenger_look的，懲罰
            - 若passenger_look  True 鼓勵taxi移動和做pick up動作
                ?過程中記錄移動過的方向 判斷移動後若passenger_look  False 鼓勵taxi移動回來 並且往未移動過的方向走 直到pick up成功
            - 同上,根據destination_look鼓勵taxi走到destination
            
            '''
            shaped_reward = 0
            potential_shaping_value = 0
            shaped_reward_detail = []
            shaped_reward_detail.append(f"next_target_station {next_target_station}")
            
            for station in stations:
                d = manhattan_distance((obs[0],obs[1]), station)
                shaped_reward_detail.append(f"station {station},  dist {d}")
            if not cur_passenger_picked_up and not next_passenger_picked_up:
                if passenger_pos_view_by_taxi is None:
                    for station in sorted_stations:
                        if station not in station_visited:
                            target_station = station
                            next_target_station = station
                            break
                    shaped_reward_detail.append(f"station_visited {station_visited}")   
                    shaped_reward_detail.append(f"sorted_stations {sorted_stations}")

                    if cur_taxi_pos != next_taxi_pos: # Taxi move action
                        if next_taxi_pos in sorted_stations and next_taxi_pos not in station_visited:
                                station_visited.add(next_taxi_pos) 
                                shaped_reward += 10
                                shaped_reward_detail.append(f"station first visit")
                                sorted_stations = sort_stations_by_distance((next_taxi_pos[0],next_taxi_pos[1]), stations)
                                for station in sorted_stations:
                                    if station not in station_visited:
                                        next_target_station = station
                                        shaped_reward_detail.append(f"next_target_station_1_vi {next_target_station}")
                                        break
                                if next_passenger_look: # if passenger in the station
                                    passenger_pos_view_by_taxi = next_taxi_pos # record passenger pos
                                    next_target_station = passenger_pos_view_by_taxi
                                    # print("record passenger pos visited")
                                    shaped_reward_detail.append(f"next_target_station_1_pa {next_target_station}")
                                elif next_destination_look:
                                    destination_pos_view_by_taxi = next_taxi_pos

                                if len(station_visited) == 3:
                                    remaining_station = [s for s in stations if s not in station_visited][0]
                                    
                                    if passenger_pos_view_by_taxi is None:
                                        passenger_pos_view_by_taxi = remaining_station
                                        next_target_station = passenger_pos_view_by_taxi
                                        shaped_reward_detail.append(f"next_target_station_1_re {next_target_station}")
                                    elif destination_pos_view_by_taxi is None:
                                        destination_pos_view_by_taxi = remaining_station

                        # elif next_taxi_pos in stations and next_taxi_pos in station_visited: # re visit, punish
                        #     shaped_reward -= 13
                        #     shaped_reward_detail.append(f"station repeat visit, -3")
                else:
                    target_station = passenger_pos_view_by_taxi
                    # print(f"passenger_pos_view_by_taxi {passenger_pos_view_by_taxi}")
                    # print(f"target_station updated 12")
                    next_target_station = passenger_pos_view_by_taxi
                    shaped_reward_detail.append(f"next_target_station_2 {next_target_station}")

                # else:
                #     potential_shaping_value = potential_shaping(cur_taxi_pos, next_taxi_pos, passenger_dropped_pos)
                #     shaped_reward_detail.append(f"go to passenger dropped pos ,potential {potential_shaping_value}")
                    # prev_distance = manhattan_distance(cur_taxi_pos,passenger_dropped_pos)
                    # next_distance = manhattan_distance(next_taxi_pos,passenger_dropped_pos)

                    # distance_change = prev_distance - next_distance
                    # shaped_reward += distance_change *2

            
            elif not cur_passenger_picked_up and  passenger_pos_view_by_taxi is not None: 
                if next_passenger_picked_up: # pick up passenger successfully
                    shaped_reward += 30 - pickup_time*3
                    pickup_time += 1 # avoid pick up repeatly
                    # potential_shaping_value = potential_shaping(cur_taxi_pos, next_taxi_pos, cur_taxi_pos)
                    shaped_reward_detail.append(f"pickup successfully,  time:{pickup_time}, reward:{shaped_reward}")
                    target_station = passenger_pos_view_by_taxi
                    # print(f"target_station updated 11")
                    # print(f"target_station {target_station}")

                    if destination_pos_view_by_taxi is None:
                        sorted_stations = sort_stations_by_distance((next_taxi_pos[0],next_taxi_pos[1]), stations)
                        for station in sorted_stations:
                            if station not in station_visited:
                                next_target_station = station
                                break
                    else:
                        next_target_station = destination_pos_view_by_taxi
                    shaped_reward_detail.append(f"next_target_station_3 {next_target_station}")
                    
                    # done = True
                elif cur_taxi_pos == passenger_pos_view_by_taxi and action == 4 and not next_passenger_picked_up:
                    # pick up on correct pos but failed
                    # passenger_pos_view_by_taxi is wrong
                    neighbors = [
                        (cur_taxi_pos[0] - 1, cur_taxi_pos[1]),
                        (cur_taxi_pos[0] + 1, cur_taxi_pos[1]),
                        (cur_taxi_pos[0], cur_taxi_pos[1] - 1),
                        (cur_taxi_pos[0], cur_taxi_pos[1] + 1)
                    ]
                    for n in neighbors:
                        if n in stations and n != destination_pos_view_by_taxi:
                            passenger_pos_view_by_taxi = n
                            break
                    else:
                        sorted_stations = sort_stations_by_distance((next_taxi_pos[0],next_taxi_pos[1]), stations)
                        for station in sorted_stations:
                            if station not in station_visited:
                                next_target_station = station
                                shaped_reward_detail.append(f"next_target_station_fix_pa_pos {next_target_station}")
                                break



            # passenger here, go to destination
            else: 
                if not next_passenger_picked_up: # passenger incorrect dropped
                    passenger_pos_view_by_taxi = cur_taxi_pos
                    next_target_station = passenger_pos_view_by_taxi
                    shaped_reward -= 25
                    shaped_reward_detail.append(f"dropped passenger before arrive! ,shaped_reward {shaped_reward}")
                if destination_pos_view_by_taxi is None:
                    for station in sorted_stations:
                        if station not in station_visited:
                            next_target_station = station
                            break
                if cur_taxi_pos != next_taxi_pos: # Taxi move action
                        if next_taxi_pos in sorted_stations and next_taxi_pos not in station_visited:
                                station_visited.add(next_taxi_pos) 
                                shaped_reward += 10
                                shaped_reward_detail.append(f"station first visit")
                                sorted_stations = sort_stations_by_distance((next_taxi_pos[0],next_taxi_pos[1]), stations)
                                for station in sorted_stations:
                                    if station not in station_visited:
                                        next_target_station = station
                                if next_destination_look:
                                    destination_pos_view_by_taxi = next_taxi_pos
                                    next_target_station = destination_pos_view_by_taxi

                                if len(station_visited) == 3:
                                    remaining_station = [s for s in stations if s not in station_visited][0]
                                    
                                    if destination_pos_view_by_taxi is None:
                                        destination_pos_view_by_taxi = remaining_station 

                        # elif next_taxi_pos in stations and next_taxi_pos in station_visited: # re visit, punish
                        #     shaped_reward -= 13
                        #     shaped_reward_detail.append(f"station repeat visit, -3")
                elif cur_taxi_pos == destination_pos_view_by_taxi and action == 5 and reward < 49 :
                    # drop but fail
                    # passenger_pos_view_by_taxi is wrong
                    neighbors = [
                        (cur_taxi_pos[0] - 1, cur_taxi_pos[1]),
                        (cur_taxi_pos[0] + 1, cur_taxi_pos[1]),
                        (cur_taxi_pos[0], cur_taxi_pos[1] - 1),
                        (cur_taxi_pos[0], cur_taxi_pos[1] + 1)
                    ]
                    for n in neighbors:
                        if n in stations and n != passenger_pos_view_by_taxi:
                            destination_pos_view_by_taxi = n
                            break
                    else:
                        destination_pos_view_by_taxi = None
                    
                shaped_reward_detail.append(f"next_target_station_4 {next_target_station}")

            # potential_shaping_value = potential_shaping(cur_taxi_pos, next_taxi_pos, target_station)
            # print(f"target_station={target_station}")

            prev_distance = manhattan_distance(cur_taxi_pos,target_station)
            next_distance = manhattan_distance(next_taxi_pos,target_station)
            distance_change = prev_distance - next_distance
            if distance_change > 0:
                shaped_reward += distance_change 
            else:
                shaped_reward += distance_change*3

            shaped_reward_detail.append(f"Distance reward: target_station={target_station}, distance_change={distance_change}, reward={shaped_reward}")
            shaped_reward_detail.append(f"passenger_pos_view_by_taxi {passenger_pos_view_by_taxi}")
            shaped_reward_detail.append(f"destination_pos_view_by_taxi {destination_pos_view_by_taxi}")
            shaped_reward_detail.append(f"cur_taxi_pos={cur_taxi_pos}, next_taxi_pos={next_taxi_pos}")
            shaped_reward_detail.append(f"prev_distance={prev_distance}, next_distance={next_distance}")
            shaped_reward_detail.append(f"target_station={target_station}, next_target_station={next_target_station}")
            shaped_reward_detail.append(f"station_visited {station_visited}")   
            shaped_reward_detail.append(f"sorted_stations {sorted_stations}")
            shaped_reward_detail.append(f"state {state}")
            # go to find passenger
            
            # Update total reward.
            reward += shaped_reward
            total_reward += reward 

            # Update q_table

            next_state =  env.get_state_by_obs(next_obs, next_target_station, passenger_pos_view_by_taxi, destination_pos_view_by_taxi)
            
            shaped_reward_detail.append(f"next_state {next_state}")
            if next_state not in q_table:
                q_table[next_state] = np.zeros(action_size)
                
            # Apply Q-learning update rule (Bellman equation).
            q_table[state][action] += alpha * ( reward + gamma * np.max(q_table[next_state]) - q_table[state][action] )
            # TODO: potentail shaping
            # q_table[state][action] += alpha * ( reward + potential_shaping_value + gamma * np.max(q_table[next_state]) - q_table[state][action] )
            

            # Move to the next state.
        
            state = next_state
            obs = next_obs

            episode_step += 1
            if debug and (episode + 1) % 500 == 0:
                env.draw_env_txt((obs[0], obs[1]),
                            action=action, step=episode_step, fuel=env.current_fuel, episode=episode+1,
                            potential=potential_shaping_value, reward=reward, shaped_reward_detail=shaped_reward_detail)
            

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

 
    return q_table, rewards_per_episode

q_table, rewards = tabular_q_learning_taxi(episodes=10000, debug=True, 
                                           alpha=0.1, gamma=0.99, decay_rate=0.9995, fuel_limit=100)
with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(q_table), f)
# plt.plot(rewards)
# plt.xlabel("Episodes")
# plt.ylabel("Total Reward")
# plt.title("Reward Shaping Training Progress")
# plt.show()