import argparse
from utils.CVRPGraph import CVRPGraph, save_results
from algorithms.greedy import CVRPNearestNeighbor
from algorithms.ACS import AntColonyCVRP

def main():
    # parser = argparse.ArgumentParser(description="Load and process CVRP data")
    # parser.add_argument("file", type=str, help="Path to the VRP file")
    # args = parser.parse_args()

    # Load the VRP data into the graph
    cvrp_graph = CVRPGraph()
    cvrp_graph.load_vrp_data("../data/A-n32-k5.vrp", 5, 300)

    # Print some basic info
    # print("Depot:", cvrp_graph.get_depot())
    # print("Vehicle Capacity:", cvrp_graph.get_capacity())
    # print("Demands:", cvrp_graph.get_demands())
    # print("Graph Nodes:", cvrp_graph.get_graph().nodes(data=True))
    # print("Graph Edges:", list(cvrp_graph.get_graph().edges(data=True)))

    # Uruchomienie algorytmu zachłannego (Nearest Neighbor)
    solver_greedy = CVRPNearestNeighbor(cvrp_graph)
    routes_greedy, total_cost_greedy = solver_greedy.solve()
    solver_greedy.display_routes()

    # Algorytm mrówkowy
    ant_solver = AntColonyCVRP(cvrp_graph)
    routes_aco, total_cost_aco = ant_solver.solve()

    # save_results(routes_greedy, total_cost_greedy, routes_aco, total_cost_aco)

    print("Greedy solution:", routes_greedy, "Cost:", total_cost_greedy)
    print("ACO solution:", routes_aco, "Cost:", total_cost_aco)

if __name__ == "__main__":
    main()