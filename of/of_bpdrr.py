import pandas as pd
import numpy as np

#FUNCTIONALITY
def system_functionality(demand, demand_nodes, expected):

    # Total delivered demand at all demand nodes
    Qi_total = demand[demand_nodes].sum(axis=1)

    # Total expected demand
    DQi_total = expected.sum(axis=1)

    # Equation 7
    functionality = 100 * Qi_total / DQi_total

    # Keep functionality values below 100  
    functionality = functionality.clip(lower=0, upper=100)

    return functionality

# 1. OF: Time without supply for hospital/firefighting (FH)
def OF_FH(supply_ratio, dtm, ratio=0.5):

    hospital_nodes = ["32941", "43816"]

    fire_nodes = ["54537", "21641"]

    # Critical nodes (hospitals and fire stations)
    critical_nodes = hospital_nodes + fire_nodes

    FH_undersupplied = supply_ratio[critical_nodes] <= ratio

    FH_value = FH_undersupplied.sum().sum() * dtm

    return FH_value, FH_undersupplied

# 2. rapidity of recovery (t95)
def OF_t95(demand, demand_nodes, expected, dtm, threshold=95, stability_hours=24):

    # System functionality
    Qi_total = demand[demand_nodes].sum(axis=1)
    DQi_total = expected.sum(axis=1)

    functionality = 100 * Qi_total / DQi_total
    functionality = functionality.clip(lower=0, upper=100)

    values = functionality.to_numpy()
    times = functionality.index.to_numpy()

    # Number of timesteps corresponding to the stability period
    stability_steps = int((stability_hours * 60) / dtm)

    # Search only where a full stability window still fits
    for i in range(len(values) - stability_steps + 1):

        if (
            values[i] >= threshold
            and np.all(values[i:i + stability_steps] >= threshold)
        ):
            return times[i] / 60      # minutes

    return np.nan

# 3. OF: Resilience loss (RL)
def OF_RL(demand, demand_nodes, expected, dtm):

    # Total demand at all demand nodes over time
    Qi_total = demand[demand_nodes].sum(axis=1)

    # Total expected demand (base demand) at all demand nodes
    DQi_total = expected.sum(axis=1)

    # Formula to calculate functionality
    functionality = 100 * Qi_total / DQi_total

    functionality = functionality.clip(lower=0, upper=100)


    loss = 100 - functionality

    resilience_loss = loss.sum() * dtm

    return resilience_loss

# 4. OF: Average time of no user service (Time no serv.)
def OF_time_no_serv(supply_ratio, dtm, ratio=0.5):

    undersupplied = supply_ratio <= ratio

    #Number of demand nodes
    DN = supply_ratio.shape[1]

    time_no_service = undersupplied.sum().sum() * dtm / DN

    return time_no_service

# 5. OF: Number of users without service for eight consecutive hours (NWSECH)
def OF_NWSECH(supply_ratio, dtm, ratio=0.5, hours=8):

    undersupplied = supply_ratio <= ratio

    # Number of required steps based on an 8-hour period and the hydraulic timestep defined in the water network model.
    threshold_steps = int((hours * 60) / dtm)

    # list to store nodes that have no service for 8 consecutive hours
    nodes_no_service = 0


    for node in undersupplied.columns:

        consecutive = 0
        max_consecutive = 0

        for value in undersupplied[node]:

            if value:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        if max_consecutive >= threshold_steps:
            nodes_no_service += 1

    return nodes_no_service

# 6. OF: Water loss (WL)
def OF_WL(demand, emitter_nodes, dt):


    leakage = demand[emitter_nodes]

    water_loss = leakage.sum().sum() * dt / 1000

    return water_loss