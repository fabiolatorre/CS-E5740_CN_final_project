import networkx as nx
import numpy as np
import random
from si_animator import visualize_si, plot_network_usa
import matplotlib as mpl
import matplotlib.pylab as plt
from sklearn.preprocessing import minmax_scale
from scipy.stats import spearmanr


def merge_dicts(dict1, dict2):
    """
    Merge dictionaries and keep values of common keys in a 1-dimensional list
    """
    result = {**dict2, **dict1}
    for key, value in result.items():
        if key in dict1 and key in dict2:
            if type(value) is not list:
                result[key] = [value, dict2[key]]
            else:
                result[key].append(dict2[key])
    return result


def compute_prevalence(n_bins, infection_list, num_nodes):
    prevalence_list = []
    curr_tot = 1
    for idx in range(n_bins):
        count = np.count_nonzero(infection_list == idx + 1)
        prevalence_list.append((count + curr_tot) / num_nodes)
        curr_tot += count
    return prevalence_list


def sample_random_neighbour(net: nx.Graph, n_samples=10):
    samples = []
    while len(samples) < n_samples:
        rand_node = np.random.choice(net.nodes)
        rand_neighbor = np.random.choice(list(net.neighbors(rand_node)))
        if int(rand_neighbor) not in samples:
            samples.append(int(rand_neighbor))
    return samples


def investigate_p(event_data, n_nodes, probs, min_ts=1229231100, max_ts=1230128400, iterations=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    for p in probs:
        prevalence_mat = []
        for i in range(iterations):
            infection_list = simulate_si(event_data=event_data,
                                         seed=0,
                                         p=p,
                                         ts_intervals=ts_bins)
            prevalence_list = compute_prevalence(n_bins, infection_list, n_nodes)
            prevalence_mat.append(prevalence_list)

        plt.plot(ts_bins, np.mean(prevalence_mat, 0), '.-', linewidth=1.5, label='p={:.2f}'.format(p))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different probabilities')
    ax.legend(loc=0)

    return fig


def investigate_seed(event_data, n_nodes, seeds, min_ts=1229231100, max_ts=1230128400, iterations=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    for seed in seeds:
        prevalence_mat = []
        for i in range(iterations):
            infection_list = simulate_si(event_data=event_data,
                                         seed=seed,
                                         p=.1,
                                         ts_intervals=ts_bins)
            prevalence_list = compute_prevalence(n_bins, infection_list, n_nodes)
            prevalence_mat.append(prevalence_list)

        plt.plot(ts_bins, np.mean(prevalence_mat, 0), '.-', linewidth=1.5, label='seed={}'.format(seed))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different seeds')
    ax.legend(loc=0)

    return fig


def investigate_immunity(net: nx.Graph, event_data, immune_nodes_dict, min_ts=1229231100, max_ts=1230128400):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    # Compute seeds
    immune_nodes_set = set(np.array([value for _, value in immune_nodes_dict.items()]).flatten())
    seeds = []
    while len(seeds) < 20:
        node = np.random.choice(net.nodes)
        if node not in immune_nodes_set:
            seeds.append(int(node))

    for strategy in immune_nodes_dict.keys():
        prevalence_mat = []
        for seed in seeds:
            infection_list = simulate_si(event_data=event_data,
                                         seed=seed,
                                         p=.5,
                                         ts_intervals=ts_bins,
                                         immune_nodes=immune_nodes_dict[strategy])
            prevalence_list = compute_prevalence(n_bins, infection_list, n_nodes)
            prevalence_mat.append(prevalence_list)

        plt.plot(ts_bins, np.mean(prevalence_mat, 0), '.-', linewidth=1.5, label='{}'.format(strategy))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different immunity strategies')
    ax.legend(loc=0)

    return fig


def compute_immune_nodes(net: nx.Graph, n_immune_nodes):

    immune_nodes = {
        "Social Network": sample_random_neighbour(net=net, n_samples=n_immune_nodes),
        "Random Node": list(np.random.choice([int(node) for node in net.nodes], n_immune_nodes))
    }

    net_stats_dict = compute_net_stats_dict(net)

    for strategy in list(net_stats_dict.keys()):
        strategy_dict = {int(k): v for k, v in dict(net_stats_dict[strategy]).items()}
        top_nodes = sorted(strategy_dict.keys(), key=lambda key: strategy_dict[key], reverse=True)[:n_immune_nodes]
        top_nodes = list(map(int, top_nodes))

        immune_nodes[strategy] = top_nodes

    return immune_nodes


def compute_median_infection_times(net: nx.Graph, event_data, p=.5, iterations=50):
    infection_dict_global = {}
    for _ in range(iterations):
        rand_seed = int(np.random.choice(net.nodes))
        infection_dict = simulate_si(event_data=event_data, seed=rand_seed, p=p, return_dict=True)
        infection_dict_global = merge_dicts(infection_dict_global, infection_dict)
    median_dict = {}
    for key, value in sorted(infection_dict_global.items()):
        median_dict[key] = int(np.median(np.array(value).flatten()))

    return median_dict


def compute_net_stats_dict(net: nx.Graph):
    stats_dict = {
        'Unweighted Clustering Coefficient': nx.clustering(net),
        'Degree': dict(net.degree),
        'Strength': net.degree(weight='weight'),
        'Unweighted Betweenness Centrality': nx.betweenness_centrality(net)
    }
    return stats_dict


def plot_statistics(net: nx.Graph, event_data, p=.5, iterations=50):

    median_infection_times = compute_median_infection_times(net, event_data, p, iterations)

    net_stats_dict = compute_net_stats_dict(net)

    for (stat_name, stat_values) in net_stats_dict.items():
        x = [stat_values[node] for node in list(net.nodes)]
        y = [median_infection_times[int(node)] for node in list(net.nodes)]

        plt.xlabel('{}'.format(stat_name))
        plt.ylabel('Normalized Median Infection Time')
        plt.scatter(x, minmax_scale(y))
        plt.title('Median Infect Time as function of {}'.format(stat_name))
        plt.savefig('{}.pdf'.format(stat_name.replace(" ", "_")))
        plt.close()

        spearman_coeff = spearmanr(x, y).correlation
        print('Spearman rank-correlation coefficient of {}: {:.5f}'.format(stat_name, spearman_coeff))


def simulate_si(event_data, seed=0, p=.5, ts_intervals=None,
                create_animation=False, return_dict=False,
                immune_nodes=[], links_stats=False):

    # Set the seed of the infection in the dict
    infection_times = {seed: event_data[0]["StartTime"]}
    links_dict = {}

    if not links_stats:
        for row in event_data:
            if not row["Destination"] in immune_nodes and row["Source"] in infection_times and\
                    random.random() <= p and infection_times[row["Source"]] <= row["StartTime"]:
                if row["Destination"] in infection_times:
                    infection_times[row["Destination"]] = min(infection_times[row["Destination"]], row["EndTime"])
                else:
                    infection_times[row["Destination"]] = row["EndTime"]
    else:  # TODO: complete links handling
        for row in event_data:
            if not row["Destination"] in immune_nodes and row["Source"] in infection_times and\
                    random.random() <= p and infection_times[row["Source"]] <= row["StartTime"]:
                if row["Destination"] in infection_times:
                    infection_times[row["Destination"]] = min(infection_times[row["Destination"]], row["EndTime"])
                else:
                    infection_times[row["Destination"]] = row["EndTime"]

    # Generate animation
    infection_list = list(infection_times.values())
    infection_list.sort()
    infection_list = np.array(infection_list)

    if create_animation:
        visualize_si(infection_list, fps=5, save_fname="anim.html")

    if return_dict:
        return infection_times

    if links_stats:
        return links_dict

    if ts_intervals is None:
        return infection_list
    else:
        return np.digitize(infection_list, ts_intervals)


if __name__ == "__main__":
    net: nx.Graph = nx.read_weighted_edgelist("./aggregated_US_air_traffic_network_undir.edg")
    ndtype = [("Source", int), ("Destination", int), ("StartTime", int), ("EndTime", int)]
    event_data = np.genfromtxt('events_US_air_traffic_GMT.txt', names=True, dtype=ndtype)
    event_data.sort(order=["StartTime"])

    n_nodes = net.number_of_nodes()
    min_ts = min(event_data[:]["StartTime"])
    max_ts = max(event_data[:]["EndTime"])

    # Task 1 ==============================================
    # print("Started first simulation")
    # infection_dict = simulate_si(event_data=event_data, seed=0, return_dict=True)
    # print("Anchorage infected at: {}".format(infection_dict[41]))

    # # Task 2 ==============================================
    # print("Started probability investigation")
    # probs = [0.01, 0.05, 0.1, 0.5, 1.0]
    #
    # fig = investigate_p(event_data=event_data,
    #                     n_nodes=n_nodes,
    #                     probs=probs,
    #                     iterations=10,
    #                     min_ts=min_ts,
    #                     max_ts=max_ts)
    #
    # fig.savefig('./p_investigation.pdf')
    # fig.clear()
    #
    # # Task 3 ==============================================
    # print("Started seed investigation")
    # seeds = [0, 4, 41, 100, 200]
    #
    # fig = investigate_seed(event_data=event_data,
    #                        n_nodes=n_nodes,
    #                        seeds=seeds,
    #                        iterations=10,
    #                        min_ts=min_ts,
    #                        max_ts=max_ts)
    #
    # fig.savefig('./seed_investigation.pdf')
    # fig.clear()
    #
    # # Task 4 ==============================================
    # plot_statistics(net=net, event_data=event_data, p=.5, iterations=50)
    #
    # Task 5 ==============================================

    n_immune_nodes = 10

    immune_nodes = compute_immune_nodes(net, n_immune_nodes)
    fig = investigate_immunity(net, event_data, immune_nodes)

    fig.savefig('./immunity_investigation.pdf')
    fig.clear()

    # Task 6 ==============================================

    # Bonus Task 1 ========================================

    # Task 7 ==============================================

    # Bonus Task 2 ========================================

