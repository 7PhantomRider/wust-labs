import csv
import networkx as nx
import matplotlib.pyplot as plt
import random

def load_github_graph():
    G = nx.Graph()
    ml_users = {}

    print("wczytywanie wierzchołków (ML)...")
    with open("musae_git_target.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ml_target"] == "1":
                ml_users[int(row["id"])] = row["name"]

    print("wczytywanie krawędzi...")
    with open("musae_git_edges.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u, v = int(row["id_1"]), int(row["id_2"])
            if u in ml_users and v in ml_users:
                G.add_node(u, name=ml_users[u])
                G.add_node(v, name=ml_users[v])
                G.add_edge(u, v, capacity=1) # przepustowość dla max flow-a

    print(f"graf ma {G.number_of_nodes()} węzłów i {G.number_of_edges()} krawędzi\n")
    return G

def show_shortest_path(G, u, v):
    try:
        path = nx.shortest_path(G, source=u, target=v)
        names = [G.nodes[node]['name'] for node in path]
        print(f"najkrótsza ścieżka: {' -> '.join(names)} (długość: {len(path)-1})")
        
        # ścieżka
        sub_g = G.subgraph(path)
        pos = nx.spring_layout(sub_g)
        plt.figure(figsize=(8, 5))
        plt.title("najkrótsza ścieżka")
        nx.draw(sub_g, pos, with_labels=True, labels=nx.get_node_attributes(sub_g, 'name'), 
                node_color='lightblue', edge_color='red', node_size=2000, font_size=10)
        plt.show()

    except nx.NetworkXNoPath:
        print("brak połączenia między tymi użytkownikami")

def check_euler(G):
    print(f"czy CAŁY graf jest eulerowski? {nx.is_eulerian(G)}")
    
    # podgraf
    try:
        cycle_edges = nx.find_cycle(G)
        sub_g = G.edge_subgraph(cycle_edges)
        
        print(f"wydzielono mały podgraf (cykl). czy jest eulerowski? {nx.is_eulerian(sub_g)}")
        euler_path = list(nx.eulerian_circuit(sub_g))
        print("przykładowa ścieżka Eulera na podgrafie:")
        print(euler_path)
        
        # wizualizacja podgrafu
        pos = nx.circular_layout(sub_g)
        plt.figure(figsize=(6, 6))
        plt.title("podgraf eulerowski")
        nx.draw(sub_g, pos, with_labels=True, node_color='lightgreen', node_size=1000)
        plt.show()

    except nx.NetworkXNoCycle:
        print("brak cykli w grafie")

def calculate_max_flow(G, u, v):
    # networkx sprawdza wartość 'capacity' na krawędziach
    flow_value, flow_dict = nx.maximum_flow(G, u, v)
    print(f"max flow między {G.nodes[u]['name']} a {G.nodes[v]['name']} wynosi: {flow_value}")

def main():
    G = load_github_graph()
    
    nodes_list = list(G.nodes())    
    random_nodes = random.sample(nodes_list, 3)
    
    periphery_nodes = [n for n, d in G.degree() if d == 1]
    random_periphery = random.sample(periphery_nodes, min(2, len(periphery_nodes)))

    print("losoew ID:")
    for node_id in random_nodes + random_periphery:
        print(f" - {node_id} ({G.nodes[node_id]['name']}) | stopień: {G.degree[node_id]}")

    while True:
        print("\n--- MENU ---")
        print("1. Najkrótsza ścieżka (BFS)")
        print("2. Sprawdzenie Eulera i wizualizacja podgrafu")
        print("3. Maksymalny przepływ")
        print("0. Wyjście")
        
        choice = input("Wybór: ")
        
        if choice == '1':
            u = int(input("start ID: "))
            v = int(input("koniec ID: "))
            if u in G and v in G:
                show_shortest_path(G, u, v)
            else:
                print("fake ID")
                
        elif choice == '2':
            check_euler(G)
            
        elif choice == '3':
            u = int(input("start ID: "))
            v = int(input("koniec ID: "))
            if u in G and v in G:
                calculate_max_flow(G, u, v)
            else:
                print("fake ID")
                
        elif choice == '0':
            break

if __name__ == "__main__":
    main()