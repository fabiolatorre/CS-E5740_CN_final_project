import networkx as nx
import matplotlib as mpl
import numpy as np
import random
from si_animator import visualize_si


def simulate_si(seed, p=1., start_ts=None, end_ts=None):
    # net = nx.read_weighted_edgelist("./aggregated_US_air_traffic_network_undir.edg")
    event_data = np.genfromtxt('events_US_air_traffic_GMT.txt', names=True, dtype=int)
    event_data.sort(order=["StartTime"])

    infection_times = {seed: event_data[0]["StartTime"]}

    for row in event_data:
        if row["Source"] in infection_times and random.random() <= p:
            if infection_times[row["Source"]] <= row["StartTime"]:
                if row["Destination"] in infection_times:
                    infection_times[row["Destination"]] = min(infection_times[row["Destination"]], row["EndTime"])
                else:
                    infection_times[row["Destination"]] = row["EndTime"]

    infection_list = list(infection_times.values())
    infection_list.sort()

    print("Anchorage infected at: {}".format(infection_times[41]))

    visualize_si(np.array(infection_list), infection_list[0], infection_list[-1], save_fname="anim.html", fps=1)


# T1
simulate_si(seed=0)

# T2
# for prob in [0.01, 0.05, 0.1, 0.5, 1.0]:
#     simulate_si(seed=0, p=prob)


