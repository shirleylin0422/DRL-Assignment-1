# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
def get_state_by_obs(obs, target_pos, passenger_pos_view_by_taxi, destination_pos_view_by_taxi):
        
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

def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def sort_stations_by_distance(cur_taxi_pos, stations):

        sorted_stations = sorted(stations, key=lambda s: manhattan_distance(cur_taxi_pos, s))
        return sorted_stations

    
target_station = None
passenger_pos_view_by_taxi = None
destination_pos_view_by_taxi = None
passenger_picked_up = False
station_visited = set()
sorted_stations = []
init = True

def get_action(obs):
    global target_station
    global passenger_pos_view_by_taxi
    global destination_pos_view_by_taxi
    global passenger_picked_up
    global station_visited
    global init
    global sorted_stations
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    taxi_pos = obs[0],obs[1]
    stations = [(obs[2],obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    passenger_look = obs[14]
    destination_look = obs[15]
    if init:
        sorted_stations = sort_stations_by_distance((obs[0],obs[1]), stations)
        init = False
    if len(station_visited) == 3:
        remaining_station = [s for s in stations if s not in station_visited][0]
        if passenger_pos_view_by_taxi is None:
            passenger_pos_view_by_taxi = remaining_station
        elif destination_pos_view_by_taxi is None:
            destination_pos_view_by_taxi = remaining_station
    
    if not passenger_picked_up:
        for station in sorted_stations:
            if station not in station_visited:
                target_station = station
                break
        if passenger_pos_view_by_taxi is None:
            if taxi_pos in stations and taxi_pos not in station_visited:
                    station_visited.add(taxi_pos) 
                    sorted_stations = sort_stations_by_distance(taxi_pos, stations)
                    for station in sorted_stations:
                        if station not in station_visited:
                            target_station = station
                            break
                    if passenger_look:
                        passenger_pos_view_by_taxi = taxi_pos
                    elif destination_look:
                        destination_pos_view_by_taxi = taxi_pos
            
        elif passenger_pos_view_by_taxi is not None and passenger_picked_up is False:
            target_station = passenger_pos_view_by_taxi
    else:
        if destination_pos_view_by_taxi is None:
            sorted_stations = sort_stations_by_distance(taxi_pos, stations)
            for station in sorted_stations:
                if station not in station_visited:
                    target_station = station
                    break
            if taxi_pos in stations and taxi_pos not in station_visited:
                    station_visited.add(taxi_pos) 
                    sorted_stations = sort_stations_by_distance(taxi_pos, stations)
                    if destination_look:
                        destination_pos_view_by_taxi = taxi_pos
                    else:
                        for station in sorted_stations:
                            if station not in station_visited:
                                target_station = station
                                break
        else:
            target_station = destination_pos_view_by_taxi

    state =  get_state_by_obs(obs, target_station, passenger_pos_view_by_taxi, destination_pos_view_by_taxi)
    # print(f"state {state}")
    if state in q_table:
        action = int(np.argmax(q_table[state]))
        if taxi_pos == passenger_pos_view_by_taxi and not passenger_picked_up and action == 4:
            passenger_picked_up = True
        if passenger_picked_up and action == 5:
            passenger_pos_view_by_taxi = taxi_pos
            passenger_picked_up = False
        return action
    else:
        return random.choice([0,1,2,3,4,5])
