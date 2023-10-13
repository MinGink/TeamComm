import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os


def measure_strength(G, node_i, node_j):
    list1 = set(G.neighbors(node_i))
    list1.add(node_i)
    list2 = set(G.neighbors(node_j))
    list2.add(node_j)
    strength =  len(list1 & list2) / ((len(list1)  * len(list2)) ** 0.5)
    return strength




def graph_test(threshold):
    G = nx.binomial_graph(10, 0.5, seed=666 , directed=False)

    g_1 = nx.Graph()
    g_1.add_nodes_from(G.nodes(data=False))

    g_2 = nx.Graph()
    g_2.add_nodes_from(G.nodes(data=False))


    g_3 = nx.Graph()
    g_3.add_nodes_from(G.nodes(data=False))

    min_max_set = set()


    for e in G.edges():
        strength = measure_strength(G, e[0], e[1])
        print(strength)
        G.edges[e[0], e[1]]['weight'] = round(strength, 2)
        min_max_set.add(strength)



    # min_max_list.sort()


    min_max_list = list(min_max_set)
    min_max_list.sort()

    # min_value = min_max_list[0]
    # max_value = min_max_list[-1]
    # thershold_1 = ((max_value - min_value) / 10) * 2 + min_value
    # thershold_2 = ((max_value - min_value) / 10) * 4 + min_value
    # thershold_3 = ((max_value - min_value) / 10) * 6 + min_value

    thershold_1 = np.percentile(np.array(min_max_list), 2 * 10)
    thershold_2 = np.percentile(np.array(min_max_list), 5 * 10)
    thershold_3 = np.percentile(np.array(min_max_list), 8 * 10)


    for e in G.edges():
        strength = G.edges[e[0], e[1]]['weight']
        if strength >= thershold_1:
            g_1.add_edge(e[0], e[1], weight=round(strength, 2))

    for e in G.edges():
        strength = G.edges[e[0], e[1]]['weight']
        if strength >= thershold_2:
            g_2.add_edge(e[0], e[1], weight=round(strength, 2))

    for e in G.edges():
        strength = G.edges[e[0], e[1]]['weight']
        if strength >= thershold_3:
            g_3.add_edge(e[0], e[1], weight=round(strength, 2))


    A1 = nx.adjacency_matrix(G).todense()

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax = ax.matshow(A1) # cmap=plt.cm.Blues)
    plt.colorbar(ax, fraction=0.25)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    fig.savefig('myplot1.png')
    plt.show()


    print(A1)

    A2 = nx.adjacency_matrix(g_1).todense()
    print(A2)

    A3 = nx.adjacency_matrix(g_2).todense()
    print(A3)

    A4 = nx.adjacency_matrix(g_3).todense()
    print(A4)




    # S = [ list(c) for c in nx.connected_components(g_1)]

    fig1 = plt.figure(figsize=(2.5, 2))
    pos = nx.spring_layout(G, seed=7)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#d9ebfc')
    # edges
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='#368a73')
    nx.draw_networkx_labels(G, pos, font_size=14)
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)

    # nx.draw(G)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("on")
    plt.tight_layout()
    plt.show()
    fig1.savefig('myplot2.png')




    fig2 = plt.figure(figsize=(2.5, 2))
    pos = nx.spring_layout(g_1, seed=7)

    # nodes
    nx.draw_networkx_nodes(g_1, pos, node_size=400, node_color='#d9ebfc')

    # edges
    nx.draw_networkx_edges(g_1, pos, width=1.5, edge_color='#368a73')
    nx.draw_networkx_labels(g_1, pos, font_size=12)
    # edge weight labels
    # edge_labels = nx.get_edge_attributes(g_1, "weight")
    # nx.draw_networkx_edge_labels(g_1, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("on")
    plt.tight_layout()
    plt.show()
    fig2.savefig('myplot3.png')





    fig3 = plt.figure(figsize=(2.5, 2))
    pos = nx.spring_layout(g_2, seed=7)

    # nodes
    nx.draw_networkx_nodes(g_2, pos, node_size=400, node_color='#d9ebfc')

    # edges
    nx.draw_networkx_edges(g_2, pos, width=1.5, edge_color='#368a73')
    nx.draw_networkx_labels(g_2, pos, font_size=12)
    # edge weight labels
    # edge_labels = nx.get_edge_attributes(g_1, "weight")
    # nx.draw_networkx_edge_labels(g_1, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("on")
    plt.tight_layout()
    plt.show()
    fig3.savefig('myplot4.png')



    # plt.figure(3)
    # nx.draw(g_2, with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=300, width=2)
    # plt.show()
    #
    # plt.figure(4)
    # nx.draw(g_3, with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=300, width=2)
    # plt.show()
    print('test')


if __name__ == '__main__':
    threshold = 0.61
    graph_test(threshold)

