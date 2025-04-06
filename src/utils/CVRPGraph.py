import networkx as nx
import numpy as np
import  json


def save_results(routes_greedy, total_cost_greedy, routes_greedy_ACS, total_cost_ACS, filename="results"):
    """
    Zapisuje wyniki algorytmu do plików JSON i TXT.

    :param routes_greedy: Lista tras dla pojazdów
    :param total_cost_greedy: Całkowity koszt rozwiązania
    :param filename: Nazwa pliku wyjściowego (bez rozszerzenia)
    """
    output_data = {
        "total_cost_greedy": total_cost_greedy,
        "routes_greedy": routes_greedy,
        "total_cost_ACS": total_cost_ACS,
        "routes_greedy_ACS": routes_greedy_ACS
    }

    # Zapis do JSON
    json_filename = f"../results/{filename}.json"
    with open(json_filename, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    # Zapis do TXT
    txt_filename = f"../results/{filename}.txt"
    with open(txt_filename, "w") as txt_file:
        txt_file.write(f"Total Cost: {total_cost_greedy}\n\n")
        for i, route in enumerate(routes_greedy):
            txt_file.write(f"Vehicle {i + 1}: {' -> '.join(map(str, route))}\n")
        txt_file.write(f"Total Cost: {total_cost_ACS}\n\n")
        for i, route in enumerate(routes_greedy_ACS):
            txt_file.write(f"Vehicle {i + 1}: {' -> '.join(map(str, route))}\n")

    print(f"Results saved to: {json_filename} and {txt_filename}")


class CVRPGraph:
    def __init__(self):
        self.name = None
        self.graph = nx.Graph()
        self.depot = None
        self.capacity = None
        self.demands = {}
        self.customers = {}
        self.num_of_trucks = None
        self.max_dist = None

    def euclidean_distance(self, node1, node2):
        """Oblicza odległość euklidesową między dwoma punktami."""
        x1, y1 = node1
        x2, y2 = node2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def add_node(self, node_id, x, y, demand=0):
        self.graph.add_node(node_id, pos=(x, y), demand=demand)
        self.demands [node_id] = demand

    def add_edge(self, node1, node2, weight):
        self.graph.add_edge(node1, node2, weight=weight)

    def load_vrp_data(self, file_path, num_of_trucks, max_dist):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        section = None
        node_positions = {}

        self.num_of_trucks = num_of_trucks
        self.max_dist = max_dist

        for line in lines:
            line = line.strip()
            if line.startswith('NAME'):
                self.name = line.split(':')[1].strip()
            elif line.startswith('CAPACITY'):
                self.capacity = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                section = 'NODE_COORD_SECTION'
                continue
            elif line.startswith('DEMAND_SECTION'):
                section = 'DEMAND_SECTION'
                continue
            elif line.startswith('DEPOT_SECTION'):
                section = 'DEPOT_SECTION'
                continue
            elif line.startswith('EOF'):
                break

            if section == 'NODE_COORD_SECTION':
                parts = list(map(int, line.split()))
                node_id, x, y = parts
                self.graph.add_node(node_id, pos=(x, y))
                if node_id != 1:  # Depot ma ID 1
                    self.customers[node_id] = (x, y)
            elif section == 'DEMAND_SECTION':
                parts = line.split()
                node_id, demand = int(parts[0]), int(parts[1])
                self.graph.nodes[node_id]['demand'] = demand
                self.demands [node_id] = demand
            elif section == 'DEPOT_SECTION':
                self.depot = 1

        # Add edges with Euclidean distance as weight
        for v in self.graph.nodes:
            for u in self.graph.nodes:
                if v < u:
                    self.add_edge(v, u, self.euclidean_distance(self.graph.nodes[v]['pos'], self.graph.nodes[u]['pos']))



    def get_graph(self):
        return self.graph

    def get_depot(self):
        return self.depot

    def get_capacity(self):
        return self.capacity

    def get_demands(self):
        return self.demands

    def get_coordinates(self, node_id):
        return self.graph.nodes[node_id]['pos']

