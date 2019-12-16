import networkx as nx
import numpy as np
import random
from si_animator import visualize_si
import matplotlib as mpl
import matplotlib.pylab as plt


def investigate_p(event_data, n_nodes, probs, colors, min_ts=1229231100, max_ts=1230128400, iterations=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    for p, color in zip(probs, colors):
        infection_lists = []
        for i in range(iterations):
            infection_list = simulate_si(event_data=event_data,
                                         seed=0,
                                         p=p,
                                         ts_intervals=ts_bins)
            infected_nodes = []
            curr_tot = 1
            for idx in range(n_bins):
                count = np.count_nonzero(infection_list == idx)
                infected_nodes.append((count + curr_tot) / n_nodes)
                curr_tot += count

            infection_lists.append(infected_nodes)
        plt.plot(ts_bins, np.mean(infection_lists, 0), '{}.-'.format(color), linewidth=.7, label='p={:.2f}'.format(p))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different probabilities')
    ax.legend(loc=0)

    return fig


def investigate_seed(event_data, n_nodes, seeds, colors, min_ts=1229231100, max_ts=1230128400, iterations=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    for seed, color in zip(seeds, colors):
        infection_lists = []
        for i in range(iterations):
            infection_list = simulate_si(event_data=event_data,
                                         seed=seed,
                                         p=.1,
                                         ts_intervals=ts_bins)
            infected_nodes = []
            curr_tot = 1
            for idx in range(n_bins):
                count = np.count_nonzero(infection_list == idx)
                infected_nodes.append((count + curr_tot) / n_nodes)
                curr_tot += count

            infection_lists.append(infected_nodes)
        plt.plot(ts_bins, np.mean(infection_lists, 0), '{}.-'.format(color), linewidth=.7, label='seed={}'.format(seed))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different seeds')
    ax.legend(loc=0)

    return fig


def simulate_si(event_data, seed=0, p=1., ts_intervals=None, create_animation=False):

    # Set the seed of the infection in the dict
    infection_times = {seed: event_data[0]["StartTime"]}

    for row in event_data:
        if row["Source"] in infection_times and random.random() <= p:
            if infection_times[row["Source"]] <= row["StartTime"]:
                if row["Destination"] in infection_times:
                    infection_times[row["Destination"]] = min(infection_times[row["Destination"]], row["EndTime"])
                else:
                    infection_times[row["Destination"]] = row["EndTime"]

    # Generate animation
    infection_list = list(infection_times.values())
    infection_list.sort()
    infection_list = np.array(infection_list)

    if create_animation:
        visualize_si(infection_list, fps=1, save_fname="anim.html")

    # Task 1
    # print("Anchorage infected at: {}".format(infection_times[41]))

    if ts_intervals is None:
        return infection_list
    else:
        return np.digitize(infection_list, ts_intervals)


if __name__ == "__main__":
    net: nx.Graph = nx.read_weighted_edgelist("./aggregated_US_air_traffic_network_undir.edg")
    event_data = np.genfromtxt('events_US_air_traffic_GMT.txt', names=True, dtype=int)
    event_data.sort(order=["StartTime"])

    n_nodes = net.number_of_nodes()
    colors = ['b', 'r', 'g', 'm', 'k', 'c']

    # Task 1 ==============================================
    # simulate_si(event_data=event_data, seed=0)

    # Task 2 ==============================================
    # probs = [0.01, 0.05, 0.1, 0.5, 1.0]
    #
    # fig = investigate_p(event_data=event_data,
    #               n_nodes=n_nodes,
    #               probs=probs,
    #               colors=colors,
    #               iterations=10,
    #               min_ts=min(event_data[:]["StartTime"]),
    #               max_ts=max(event_data[:]["EndTime"]))
    #
    # fig.savefig('./p_investigation.pdf')

    # Task 3 ==============================================
    seeds = [0, 4, 41, 100, 200]

    fig = investigate_seed(event_data=event_data,
                  n_nodes=n_nodes,
                  seeds=seeds,
                  colors=colors,
                  iterations=10,
                  min_ts=min(event_data[:]["StartTime"]),
                  max_ts=max(event_data[:]["EndTime"]))

    fig.savefig('./seed_investigation.pdf')

