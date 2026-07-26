import pandas as pd
import numpy as np

#FUNCTIONALITY
def system_functionality(demand, demand_nodes, expected):

    # Total delivered demand at all demand nodes
    Qi_total = demand[demand_nodes].sum(axis=1)

    # Total expected demand
    DQi_total = sum(expected.values())

    # Equation 7
    functionality = 100 * Qi_total / DQi_total


    return functionality

# 1. OF: Time without supply for hospital/firefighting (FH)
def OF_FH(
    wn,
    demand,
    dtm,
    hospital_nodes=None,
    fire_nodes=None,
    ratio=0.5,
):

    if hospital_nodes is None:
        hospital_nodes = ["32941", "43816"]

    if fire_nodes is None:
        fire_nodes = ["54537", "21641"]

    # Critical nodes (hospitals and fire stations)
    critical_nodes = hospital_nodes + fire_nodes

    # water demand in critical nodes (base demand)
    DQfh = {
        node: wn.get_node(node).base_demand * 1000
        for node in critical_nodes
    }

    # Dataframe to store undersupplied information for each critical node over time
    FH_undersupplied = pd.DataFrame({
        node: (demand[node] / DQfh[node]) <= ratio
        for node in critical_nodes
    })

    # Results of the FH OF in minutes
    FH_value = FH_undersupplied.sum().sum() * dtm

    return FH_value, FH_undersupplied

# 2. rapidity of recovery (t95)
def OF_t95(demand, demand_nodes, expected, threshold=95):

    Qi_total = demand[demand_nodes].sum(axis=1)
    DQi_total = sum(expected.values())

    functionality = 100 * Qi_total / DQi_total


    values = functionality.to_numpy()
    times = functionality.index.to_numpy()

    for i in range(len(values)):
        if values[i] >= threshold and np.all(values[i:] >= threshold):
            return times[i]

    return np.nan

# 3. OF: Resilience loss (RL)
def OF_RL(demand, demand_nodes, expected, dtm):

    # Total demand at all demand nodes over time
    Qi_total = demand[demand_nodes].sum(axis=1)

    # Total expected demand (base demand) at all demand nodes
    DQi_total = sum(expected.values())

    # Formula to calculate functionality
    functionality = 100 * Qi_total / DQi_total


    loss = 100 - functionality

    resilience_loss = loss.sum() * dtm

    return resilience_loss

# 4. OF: Average time of no user service (Time no serv.)
def OF_time_no_serv(
    demand,
    demand_nodes,
    expected,
    dtm,
    ratio_tns=0.5
):

    # Dataframe to store undersupplied information for each node over time
    undersupplied = pd.DataFrame({
        node: (demand[node] / expected[node]) <= ratio_tns
        for node in demand_nodes
    })

    # Total number of demand nodes
    DN = len(demand_nodes)

    # Calculating time no serv. OF
    time_no_service = (undersupplied.sum().sum() * dtm) / DN

    return time_no_service

# 5. OF: Number of users without service for eight consecutive hours (NWSECH)
def OF_NWSECH(
    demand,
    demand_nodes,
    expected,
    dtm,
    ratio=0.5,
    hours=8
):

    undersupplied = pd.DataFrame({
        node: (demand[node] / expected[node]) <= ratio
        for node in demand_nodes
    })

    # Number of required steps based on an 8-hour period and the hydraulic timestep defined in the water network model.
    threshold_steps = int((hours * 60) / dtm)

    # list to store nodes that have no service for 8 consecutive hours
    nodes_no_service = 0


    for node in demand_nodes:

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
def OF_WL(
    demand,
    emitter_nodes,
    dt
):


    leakage = demand[emitter_nodes]

    water_loss = leakage.sum().sum() * dt / 1000

    return water_loss