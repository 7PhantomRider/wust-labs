import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

def load_github_graph():
    G = nx.Graph()
    ml_users = set()

    print("wczytywanie węzłów...")
    with open("musae_git_target.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ml_target"] == "1":
                ml_users.add(int(row["id"]))

    print("wczytywanie krawędzi...")
    with open("musae_git_edges.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u, v = int(row["id_1"]), int(row["id_2"])
            if u in ml_users and v in ml_users:
                G.add_edge(u, v)

    print(f"graf: {G.number_of_nodes()} węzłów, {G.number_of_edges()} krawędzi\n")
    return G

def handle_matrices(G):
    nodes = list(G.nodes())
    A = sp.csr_matrix(nx.adjacency_matrix(G, nodelist=nodes), dtype=np.int8)
    B = sp.csr_matrix(nx.incidence_matrix(G, nodelist=nodes, oriented=False), dtype=np.int8)
    sp.save_npz("adj.npz", A)
    sp.save_npz("inc.npz", B)
    np.save("nodes.npy", np.array(nodes))  # tu dzieki temu po wczytaniu mieć oryginalne id
    print(f"zapisano: sąsiedztwo {A.shape}, incydencja {B.shape}")

    # load
    A, B = sp.load_npz("adj.npz"), sp.load_npz("inc.npz")
    nodes = np.load("nodes.npy").tolist()

    # sąsiedztwo
    G_adj = nx.Graph()
    # tu macierz była symetryczna więc bez sensu było całej uzywać, sp.triu(A)
    G_adj.add_edges_from((nodes[i], nodes[j]) for i, j in zip(*sp.triu(A).nonzero()))

    # incydencja
    G_inc = nx.Graph()
    G_inc.add_edges_from((nodes[i], nodes[j]) for i, j in sp.csc_matrix(B).indices.reshape(-1, 2))

    orig = {frozenset(e) for e in G.edges()}
    for name, H in [("sąsiedztwa", G_adj), ("incydencji", G_inc)]:
        same = {frozenset(e) for e in H.edges()} == orig
        print(f"graf z macierzy {name}: {H.number_of_edges()} krawędzi, zgodny z oryginałem: {same}")

def draw_network(G):
    print("tworzenie grafu...")
    pos = nx.spring_layout(G, iterations=30, seed=42)
    plt.figure(figsize=(10, 10))
    plt.title("sieć z GitHuba (ML), kolor = stopień")
    nx.draw(G, pos, node_size=5, node_color=[d for _, d in G.degree()],
            cmap="viridis", width=0.2, edge_color="lightgray")
    plt.show()

def calculate_metrics(G):
    # gęstość
    print(f"\ngęstość: {nx.density(G):.6f}")

    # średnica największej składowej
    A = nx.adjacency_matrix(G)
    n_comp, labels = connected_components(A, directed=False)
    big = np.where(labels == np.bincount(labels).argmax())[0]
    sub = A[big][:, big]
    diam = max(shortest_path(sub, unweighted=True, directed=False,
                             indices=list(range(i, min(i + 500, len(big))))).max()
               for i in range(0, len(big), 500))
    print(f"\nskładowych: {n_comp}, największa: {len(big)} węzłów, średnica: {int(diam)}")

    # rozkład stopni
    degs = np.array([d for _, d in G.degree()])
    print(f"\nstopnie: min {degs.min()}, max {degs.max()}, średnia {degs.mean():.2f}, mediana {np.median(degs):.0f}")
    plt.figure(figsize=(7, 4))
    plt.hist(degs, bins=50, color="blue", edgecolor="white")
    plt.yscale("log")
    plt.title("rozkład stopni (oś Y logarytmiczna)")
    plt.xlabel("liczba znajomych")
    plt.ylabel("liczba osób")
    plt.show()

def main():
    G = load_github_graph()

    while True:
        print("\n--- ZADANIE 2 ---")
        print("1. macierze (zapisz, wczytaj, odtwórz)")
        print("2. wizualizacja grafu")
        print("3. metryki strukturalne")
        print("0. wyjście")

        choice = input("wybór: ")

        if choice == '1':
            handle_matrices(G)
        elif choice == '2':
            draw_network(G)
        elif choice == '3':
            calculate_metrics(G)
        elif choice == '0':
            break
        else:
            print("niepoprawny wybór")

if __name__ == "__main__":
    main()