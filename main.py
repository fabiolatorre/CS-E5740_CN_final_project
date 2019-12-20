import networkx as nx
import numpy as np
import random
import os
from si_animator import visualize_si, plot_network_usa
import matplotlib.pylab as plt
from sklearn.preprocessing import minmax_scale
from scipy.stats import spearmanr
import logging


def merge_dicts(dict1, dict2):
    """
    Merge dictionaries and keep values of common keys in a 1-dimensional list
    :param dict1: first dictionary to merge
    :param dict2: second dictionary to merge
    :return: dictionary whose keys are the union of dict1's and dict2's keys and
             the values of common keys are 1-dimensional lists containing both values
    """
    # Merge dictionaries keeping the values of dict1 for the common keys
    result = {**dict2, **dict1}

    # Generate the lists for the common keys
    for key, value in result.items():
        if key in dict1 and key in dict2:
            if type(value) is not list:
                result[key] = [value, dict2[key]]
            else:
                result[key].append(dict2[key])
    return result


def compute_prevalence(n_bins, infection_list, num_nodes):
    """
    Computes the prevalence (fraction of infected nodes) from a list of infection timestamps
    :param n_bins: number of timestamp bins
    :param infection_list: list of sorted timestammps in which infections occurred
    :param num_nodes: number of nodes of the network
    :return:
    """
    prevalence_list = []
    curr_tot = 1  # counter of the infected nodes. Initialized to 1 because of the seed
    for idx in range(n_bins):
        count = np.count_nonzero(infection_list == idx + 1)
        prevalence_list.append((count + curr_tot) / num_nodes)
        curr_tot += count
    return prevalence_list


def sample_random_neighbour(net: nx.Graph, n_samples=10):
    """
    Randomly sample neighbours of random nodes
    :param net: graph from which nodes are sampled
    :param n_samples: number of sampled nodes
    :return: list of n_samples nodes
    """
    samples = []
    while len(samples) < n_samples:
        rand_node = np.random.choice(net.nodes)
        rand_neighbor = np.random.choice(list(net.neighbors(rand_node)))
        if int(rand_neighbor) not in samples:
            samples.append(int(rand_neighbor))
    return samples


def investigate_p(event_data, n_nodes, probs, min_ts=1229231100, max_ts=1230128400, iterations=10):
    """
    Investigate the behavior of the prevalence over different probabilities of infection
    :param event_data: numpy array containing data relative to the events
    :param n_nodes: number of nodes of the considered graph
    :param probs: list of probabilities to investigate
    :param min_ts: smallest timestamp in the event list (used for computing the bins)
    :param max_ts: greatest timestamp in the event list (used for computing the bins)
    :param iterations: number of iterations for each probability
    :return: pyplot figure containing the plot for each probability
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    # Compute the average prevalence for each probability
    for p in probs:
        prevalence_mat = []
        for i in range(iterations):
            infection_list = simulate_si(event_data=event_data,
                                         seed=0,
                                         p=p,
                                         ts_intervals=ts_bins)
            prevalence_list = compute_prevalence(n_bins, infection_list, n_nodes)
            prevalence_mat.append(prevalence_list)

        plt.plot(ts_bins-min_ts, np.mean(prevalence_mat, 0), '.-', linewidth=1.5, label='p={:.2f}'.format(p))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different probabilities')
    ax.legend(loc=0)

    return fig


def investigate_seed(event_data, n_nodes, seeds, min_ts=1229231100, max_ts=1230128400, iterations=10):
    """
    Investigate the behavior of the prevalence over different infected seeds
    :param event_data: numpy array containing data relative to the events
    :param n_nodes: number of nodes of the considered graph
    :param seeds: list of seeds to investigate
    :param min_ts: smallest timestamp in the event list (used for computing the bins)
    :param max_ts: greatest timestamp in the event list (used for computing the bins)
    :param iterations: number of iterations for each seed
    :return: pyplot figure containing the plot for each seed
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    # Compute the average prevalence for each seed
    for seed in seeds:
        prevalence_mat = []
        for i in range(iterations):
            infection_list = simulate_si(event_data=event_data,
                                         seed=seed,
                                         p=.1,
                                         ts_intervals=ts_bins)
            prevalence_list = compute_prevalence(n_bins, infection_list, n_nodes)
            prevalence_mat.append(prevalence_list)

        plt.plot(ts_bins-min_ts, np.mean(prevalence_mat, 0), '.-', linewidth=1.5, label='seed={}'.format(seed))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different seeds')
    ax.legend(loc=0)

    return fig


def investigate_immunity(net: nx.Graph, event_data, immune_nodes_dict, min_ts=1229231100, max_ts=1230128400):
    """
    Investigate the behavior of the prevalence over different immunity strategies
    :param net: network to analyze
    :param event_data: numpy array containing data relative to the events
    :param immune_nodes_dict: dictionary containing the name of the strategies as keys and the list of relative immune
                              nodes as values
    :param min_ts: smallest timestamp in the event list (used for computing the bins)
    :param max_ts: greatest timestamp in the event list (used for computing the bins)
    :return: pyplot figure containing the plot for each immunity strategy
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = 50
    ts_bins = np.linspace(min_ts, max_ts, n_bins)

    # Compute random seeds among the non-immune nodes (same seeds for all immunity stategies)
    immune_nodes_set = set(np.array([value for _, value in immune_nodes_dict.items()]).flatten())
    seeds = []
    while len(seeds) < 20:
        node = np.random.choice(net.nodes)
        if node not in immune_nodes_set:
            seeds.append(int(node))

    # Compute the average prevalence for each strategy
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

        plt.plot(ts_bins-min_ts, np.mean(prevalence_mat, 0), '.-', linewidth=1.5, label='{}'.format(strategy))

    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Prevalence')
    ax.set_title(r'Prevalence over time with different immunity strategies')
    ax.legend(loc=0)

    return fig


def compute_immune_nodes(net: nx.Graph, n_immune_nodes):
    """
    Compute a dictionary containing the name of the different immunity strategies as keys and the
    list of relative immune nodes as values
    :param net: graph from which immune nodes are picked
    :param n_immune_nodes: number of immune nodes for each strategy
    :return: dictionary containing the name of the different immunity strategies as keys and the
             list of relative immune nodes as values
    """
    # Initialize the resulting dictionary with the 'Social Network' and the 'Random Node' approaches
    immune_nodes = {
        "Social Network": sample_random_neighbour(net=net, n_samples=n_immune_nodes),
        "Random Node": list(np.random.choice([int(node) for node in net.nodes], n_immune_nodes))
    }

    # Compute some network statistics as a dictionary containing the name of the statistic as a key and the value
    # for each node as value
    net_stats_dict = compute_net_stats_dict(net)

    # Keep the n_immune_nodes nodes with the greatest value from each statistic
    for strategy in list(net_stats_dict.keys()):
        strategy_dict = {int(k): v for k, v in dict(net_stats_dict[strategy]).items()}
        top_nodes = sorted(strategy_dict.keys(), key=lambda key: strategy_dict[key], reverse=True)[:n_immune_nodes]
        top_nodes = list(map(int, top_nodes))

        immune_nodes[strategy] = top_nodes

    return immune_nodes


def compute_median_infection_times(net: nx.Graph, event_data, p=.5, iterations=50):
    """
    Computes the median infection time for each node over a given number of iterations with random seeds
    :param net: network to analyze
    :param event_data: numpy array containing data relative to the events
    :param p: probability of infection
    :param iterations: number of iterations (with different seed nodes)
    :return: dictionary with nodes as keys and the relative median infection time as values
    """

    # Dictionary to contain the nodes as keys and the relative list of infection times as values
    infection_dict_global = {}

    # Fill infection_dict_global
    for _ in range(iterations):
        rand_seed = int(np.random.choice(net.nodes))
        infection_dict = simulate_si(event_data=event_data, seed=rand_seed, p=p, return_dict=True)
        infection_dict_global = merge_dicts(infection_dict_global, infection_dict)
    median_dict = {}

    # Compute the medians
    for key, value in sorted(infection_dict_global.items()):
        median_dict[key] = int(np.median(np.array(value).flatten()))

    return median_dict


def compute_net_stats_dict(net: nx.Graph):
    """
    Compute a dictionary containing the network node statistics for each node of a given network
    :param net: graph from which statistics are computed
    :return: dictionary containing the network node statistics for each node
    """
    stats_dict = {
        'Unweighted Clustering Coefficient': nx.clustering(net),
        'Degree': dict(net.degree),
        'Strength': net.degree(weight='weight'),
        'Unweighted Betweenness Centrality': nx.betweenness_centrality(net)
    }
    return stats_dict


def compute_net_links_stats_dict(net: nx.Graph):
    """
    Compute a dictionary containing the network edges statistics for each edge of a given network
    :param net: graph from which statistics are computed
    :return: dictionary containing the network node statistics for each edge
    """
    links_stats_dict = {
        'Weight': nx.get_edge_attributes(net, 'weight'),
        'Link Betweenness Centrality': nx.edge_betweenness_centrality(net)
    }
    return links_stats_dict


def plot_statistics(net: nx.Graph, event_data, p=.5, iterations=50, result_dir="."):
    """
    Given a network and an array of events, computes the median infection times of the nodes and plots it as function
    of different network statistics and prints the spearman correlation between the median time and the statistics
    :param net: network to analyze
    :param event_data: numpy array containing data relative to the events
    :param p: probability of infection
    :param iterations: number of iterations on which the median infection times are computed
    :param result_dir: directory in which plots must be stored
    """
    # Compute median infection time dictionary
    median_infection_times = compute_median_infection_times(net, event_data, p, iterations)

    # Compute network node statistics
    net_stats_dict = compute_net_stats_dict(net)

    # Plot the median infection time as function of each node statistic and save the figure
    for stat_name, stat_values in net_stats_dict.items():
        x = [stat_values[node] for node in list(net.nodes)]
        y = [median_infection_times[int(node)] for node in list(net.nodes)]

        plt.xlabel('{}'.format(stat_name))
        plt.ylabel('Normalized Median Infection Time')
        plt.scatter(x, minmax_scale(y))  # TODO: check normalization
        plt.title('Median Infect Time as function of {}'.format(stat_name))
        plt.savefig('{}/{}.pdf'.format(result_dir, stat_name.replace(" ", "_")))
        plt.close()

        spearman_coeff = spearmanr(x, y).correlation
        print('Spearman rank-correlation coefficient of {}: {:.5f}'.format(stat_name, spearman_coeff))


def plot_links_statistics(net: nx.Graph, link_weights, result_dir="."):
    """
    Given a network and a set of weights for the links, plots the weights of the edges as function of some
    network edge statistics and prints the spearman correlation between the weights and the statistics
    :param net: network to analyze
    :param link_weights: weights of the edges to be plotted as function of some edge statistics
    :param result_dir: directory in which plots must be stored
    """
    # Compute the network edge statistics
    net_links_stats_dict = compute_net_links_stats_dict(net)

    # Compute list of all the edge tuples of the network
    edges = list(net.edges())

    # Add missing edges to the the weights received as parameters
    weights = []
    for edge in edges:
        if edge in link_weights:
            weights.append(link_weights[edge])
        else:
            weights.append(0)

    # Plot the weights as function of each edge statistic and save the figure
    for stat_name, stat_values in net_links_stats_dict.items():
        x = [stat_values[edge] for edge in edges]
        y = weights

        plt.xlabel('Normalized {}'.format(stat_name))
        plt.ylabel('Transmission ratio')
        plt.scatter(minmax_scale(x), y)  # TODO: Check normalization
        plt.title('Transmission ratio as function of {}'.format(stat_name))
        plt.savefig('{}/{}.pdf'.format(result_dir, stat_name.replace(" ", "_")))
        plt.close()

        spearman_coeff = spearmanr(x, y).correlation
        print('Spearman rank-correlation coefficient of {}: {:.5f}'.format(stat_name, spearman_coeff))


def compute_transmitting_links(net: nx.Graph, event_data, p=.5, iterations=20):
    """
    Compute transmission fraction for each edge in the network that transmits the disease at least once
    :param net: network to analyze
    :param event_data: numpy array containing data relative to the events
    :param p: probability of infection
    :param iterations: number of iterations (random seed for every iteration)
    :return: dictionary with edge tuples as keys and fraction of transmission as values
    """
    # List of nodes in the network
    nodes = list(net.nodes)

    # Pick a random seed for each iteration
    seeds = np.random.choice(nodes, iterations)

    links_dictionary = {}
    # Compute the number of transmission for each edge over the iterations and fill links_dictionary()
    for seed in seeds:
        links_dictionary = merge_dicts(links_dictionary, simulate_si_links(event_data, int(seed), p))

    # Sum the links weight obtained over the iterations
    for k, v in links_dictionary.items():
        if type(v) is list:
            links_dictionary[k] = sum(v)

    # Create a copy of links_dictionary where k[0] <= k[1] and sum values of the same edge if present in
    # both directions. Divide also by the number of iterations so as to obtain fractions
    links_dictionary_sorted = {}
    for k, v in links_dictionary.items():
        if (k[1], k[0]) in links_dictionary:
            links_dictionary_sorted[(str(min(k[0], k[1])), str(max(k[0], k[1])))] =\
                (v + links_dictionary[(k[1], k[0])]) / iterations
        else:
            links_dictionary_sorted[(str(min(k[0], k[1])), str(max(k[0], k[1])))] = v / iterations

    return links_dictionary_sorted


def simulate_si_links(event_data, seed=0, p=.5):
    """
    Simulate the infection and compute a dictionary of edges that transmitted the disease at least once
    :param event_data: numpy array containing data relative to the events
    :param seed: seed of the infection
    :param p: probability of infection
    :return: dictionary containing the edges that transmitted the disease at least once as keys and the number
             of infection that passed through them (at most 1) as value
    """

    # Set the seed of the infection in the dictionary of the infected nodes
    infection_times = {seed: event_data[0]["StartTime"]}

    # Initialize the dictionary of the transmitting edges
    links_dict = {}

    # Simulate event-wise and store both the infection times and the transmitting links in the relative dictionaries
    for row in event_data:
        # If the Source is already infected and the flight starts when it was already infected
        if row["Source"] in infection_times and infection_times[row["Source"]] <= row["StartTime"]:
            # If the destination is already infected
            if row["Destination"] in infection_times:
                # If the infected flights lands before the currently registered infection time of the Destination
                if infection_times[row["Destination"]] > row["EndTime"]:
                    # Replace the infection time with the oldest
                    infection_times[row["Destination"]] = row["EndTime"]
                    # Remove previous transmission link
                    links_dict.pop(([item for item in links_dict.keys() if item[1] == row["Destination"]][0]))
                    # Add the new transmission link
                    links_dict[(row["Source"], row["Destination"])] = 1

            # If Destination not already infected
            else:
                # Check the probability
                if random.random() <= p:
                    # Register the infection time of the node and the transmission link
                    infection_times[row["Destination"]] = row["EndTime"]
                    links_dict[(row["Source"], row["Destination"])] = 1

    return links_dict


def simulate_si(event_data, seed=0, p=.5, ts_intervals=None,
                generate_animation=False, return_dict=False,
                immune_nodes=[]):
    """
    Simulate the infection according to the given parameters
    :param event_data: numpy array containing data relative to the events
    :param seed: seed of the infection
    :param p: probability of infection
    :param ts_intervals: bins to compute the binned timestamps. If set, the returned infection_list is binned
    :param generate_animation: flag indicating whether to generate the animation of the infection or not (takes time)
    :param return_dict: flag indicating whether the return must be the infection time dictionry {node: infection time}
    :param immune_nodes: list of nodes that are immune to the infection
    :return: if return_dict set, returns a dictionary with infected nodes as keys and their infection time as value;
             if ts_intervals set, returns an ordered list of binned timestamps in which the nodes have been infected;
             else, returns an ordered list of timestamps in which the nodes have been infected.
    """

    # Set the seed of the infection in the dictionary of the infected nodes
    infection_times = {seed: event_data[0]["StartTime"]}

    for row in event_data:
        # If the destination is not immune and the Source is already infected and
        # the flight starts when it was already infected
        if not row["Destination"] in immune_nodes and row["Source"] in infection_times and\
                infection_times[row["Source"]] <= row["StartTime"]:
            # If the destination is already infected
            if row["Destination"] in infection_times:
                # Keep the oldest infection time
                infection_times[row["Destination"]] = min(infection_times[row["Destination"]], row["EndTime"])

            # If Destination not already infected
            else:
                # Check the probability
                if random.random() <= p:
                    # Register the infection time of the node and the transmission link
                    infection_times[row["Destination"]] = row["EndTime"]

    # Compute ordered list of infection times
    infection_list = list(infection_times.values())
    infection_list.sort()
    infection_list = np.array(infection_list)

    # If required, generate and save the animation
    if generate_animation:
        visualize_si(infection_list, fps=5, save_fname="anim.html")

    if return_dict:
        return infection_times

    if ts_intervals is None:
        return infection_list
    else:
        return np.digitize(infection_list, ts_intervals)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    net: nx.Graph = nx.read_weighted_edgelist("./aggregated_US_air_traffic_network_undir.edg")
    ndtype = [("Source", int), ("Destination", int), ("StartTime", int), ("EndTime", int)]
    event_data = np.genfromtxt('events_US_air_traffic_GMT.txt', names=True, dtype=ndtype)
    event_data.sort(order=["StartTime"])

    results_directory = "./results"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    n_nodes = net.number_of_nodes()
    min_ts = min(event_data[:]["StartTime"])
    max_ts = max(event_data[:]["EndTime"])

    # # Task 1 ==============================================
    # logging.info("[Starting] Task 1: ANC infection time computation")
    # infection_dict = simulate_si(event_data=event_data, seed=0, p=1, return_dict=True)
    # print("Anchorage infected at: {}".format(infection_dict[41]))
    # logging.info("[Finished] Taks 1")
    #
    # # Task 2 ==============================================
    # logging.info("[Starting] Task 2: probability investigation")
    # probs = [0.01, 0.05, 0.1, 0.5, 1.0]
    #
    # fig = investigate_p(event_data=event_data,
    #                     n_nodes=n_nodes,
    #                     probs=probs,
    #                     iterations=10,
    #                     min_ts=min_ts,
    #                     max_ts=max_ts)
    #
    # fig.savefig('{}/p_investigation.pdf'.format(results_directory))
    # fig.clear()
    # logging.info("[Finished] Task 2")
    #
    # # Task 3 ==============================================
    # logging.info("[Starting] Task 3: seed investigation")
    # seeds = [0, 4, 41, 100, 200]
    #
    # fig = investigate_seed(event_data=event_data,
    #                        n_nodes=n_nodes,
    #                        seeds=seeds,
    #                        iterations=10,
    #                        min_ts=min_ts,
    #                        max_ts=max_ts)
    #
    # fig.savefig('{}/seed_investigation.pdf'.format(results_directory))
    # fig.clear()
    #
    # logging.info("[Finished] Task 3")
    #
    # # Task 4 ==============================================
    # logging.info("[Starting] Task 4: computation of node statistics")
    #
    # plot_statistics(net=net, event_data=event_data, p=.5, iterations=50, result_dir=results_directory)
    #
    # logging.info("[Finished] Task 4")
    #
    # # Task 5 ==============================================
    # logging.info("[Starting] Task 5: node simulation with immune nodes")
    #
    # n_immune_nodes = 10
    #
    # immune_nodes = compute_immune_nodes(net, n_immune_nodes)
    # fig = investigate_immunity(net, event_data, immune_nodes)
    #
    # fig.savefig('{}/immunity_investigation.pdf'.format(results_directory))
    # fig.clear()
    #
    # logging.info("[Finished] Task 5")

    # Task 6 ==============================================
    logging.info("[Starting] Task 6: link transmission simulation")

    id_data = np.genfromtxt("./US_airport_id_info.csv", delimiter=',', dtype=None, names=True, encoding=None)
    xycoords = {}
    for row in id_data:
        xycoords[str(row['id'])] = (row['xcoordviz'], row['ycoordviz'])

    links_weights = compute_transmitting_links(net, event_data)

    plot_network_usa(net, xycoords, edges=list(links_weights.keys()), linewidths=list(links_weights.values()))
    plt.title("Transmission Links")
    plt.savefig('{}/plot_network_usa.pdf'.format(results_directory))
    plt.close()

    max_spanning_tree = nx.maximum_spanning_tree(net)
    max_spanning_tree_edges = list(max_spanning_tree.edges)
    plot_network_usa(max_spanning_tree, xycoords, max_spanning_tree_edges, [1 for _ in max_spanning_tree_edges])
    plt.savefig('{}/plot_max_spanning_tree.pdf'.format(results_directory))
    plt.close()

    plot_links_statistics(net, links_weights, result_dir=results_directory)

    logging.info("[Finished] Task 6")
