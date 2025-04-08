# import argparse
# from utils.CVRPGraph import CVRPGraph, save_results
# from algorithms.greedy import CVRPNearestNeighbor
# from algorithms.ACS import AntColonyCVRP
# import time
# import numpy as np
# from collections import defaultdict
# import matplotlib.pyplot as plt
#
#
# def main():
#     # parser = argparse.ArgumentParser(description="Load and process CVRP data")
#     # parser.add_argument("file", type=str, help="Path to the VRP file")
#     # args = parser.parse_args()
#
#     # Load the VRP data into the graph
#     # files = ["A-n32-k5.vrp", "A-n34-k5.vrp", "A-n36-k5.vrp", "A-n44-k6.vrp", "A-n46-k7.vrp", "A-n48-k7.vrp",
#     #         "A-n54-k7.vrp", "A-n55-k9.vrp", "A-n60-k9.vrp", "A-n62-k8.vrp", "A-n63-k10.vrp", "A-n64-k9.vrp"]
#
#     # files = ["A-n34-k5.vrp", "A-n44-k6.vrp", "A-n46-k7.vrp", "A-n48-k7.vrp",
#     #           "A-n60-k9.vrp", "A-n62-k8.vrp", "A-n63-k10.vrp"]
#     #
#     # #optimal = [784, 778, 799, 937, 914, 1073, 1167, 1073, 1354, 1288, 1314, 1401]
#     # optimal = [778, 937, 914, 1073, 1167, 1288, 1314]
#     #
#     cvrp_graph = CVRPGraph()
#     cvrp_graph.load_vrp_data("../data/A-n63-k10.vrp", 10, 300)
#     ant = AntColonyCVRP(cvrp_graph, num_iterations=300, num_ants=100)
#     route, cost = ant.solve_2_opt()
#     print("ACO solution:", route, "Cost:", cost)
#     #
#     # # graphs = []
#     # #
#     # # for file in files:
#     # #     g = CVRPGraph()
#     # #     k = file.find("k")
#     # #     g.load_vrp_data("../data/" + file, file[k+1], 400)
#     # #     graphs.append(g)
#     #
#     # results = defaultdict(dict)
#     #
#     # idx = 0
#     #
#     # for file in files:
#     #     print(f"Przetwarzanie: {file}")
#     #     k = int(file[file.find("k") + 1:file.find(".vrp")])
#     #     graph = CVRPGraph()
#     #     graph.load_vrp_data(f"../data/{file}", k, 400)
#     #
#     #     # --- 1. Greedy ---
#     #     start = time.time()
#     #     solver_greedy = CVRPNearestNeighbor(graph)
#     #     greedy_route, greedy_cost = solver_greedy.solve()
#     #     end = time.time()
#     #     results[file]["Greedy"] = {"cost": greedy_cost, "time": end - start}
#     #
#     #     # --- 2. ACO ---
#     #     ant = AntColonyCVRP(graph, num_iterations=300, num_ants=10)
#     #     start = time.time()
#     #     route, cost = ant.solve()
#     #     end = time.time()
#     #     results[file]["ACO"] = {"cost": cost, "time": end - start}
#     #
#     #     results[file]["Optimal"] = {"cost": optimal[idx], "time": 0}
#     #     idx += 1
#     #
#     #     # --- 3. ACO + 2-opt ---
#     #     ant2 = AntColonyCVRP(graph, num_iterations=300, num_ants=10)
#     #     start = time.time()
#     #     route2, cost2 = ant2.solve_2_opt()
#     #     end = time.time()
#     #     results[file]["ACO_2opt"] = {"cost": cost2, "time": end - start}
#     #
#     #     # --- 4. ACO  + Elite ---
#     #     # ant3 = AntColonyCVRP(graph, num_iterations=300, num_ants=10)
#     #     # start = time.time()
#     #     # route3, cost3 = ant3.solve_elite_ants()
#     #     # end = time.time()
#     #     # results[file]["ACO_elite"] = {"cost": cost3, "time": end - start}
#     #
#     # methods = ["ACO", "ACO_2opt", "Optimal"]
#     # colors = ["#90CAF9", "#66BB6A", "#FFCA28", "#EF5350"]
#     # x = np.arange(len(files))
#     # width = 0.2
#     #
#     # # WYKRES KOSZTÓW
#     # plt.figure(figsize=(16, 6))
#     # for i, method in enumerate(methods):
#     #     cost_values = [results[file][method]["cost"] for file in files]
#     #     plt.bar(x + i * width - width * 1.5, cost_values, width, label=method, color=colors[i])
#     # plt.xticks(x, files, rotation=45, ha='right')
#     # plt.ylabel("Całkowity koszt trasy")
#     # plt.title("Porównanie jakości rozwiązania (koszt)")
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.grid(axis='y')
#     # plt.savefig("koszty_algorytmy.png")
#     # plt.show()
#     #
#     # # WYKRES CZASU
#     # plt.figure(figsize=(16, 6))
#     # for i, method in enumerate(methods):
#     #     time_values = [results[file][method]["time"] for file in files]
#     #     plt.bar(x + i * width - width * 1.5, time_values, width, label=method, color=colors[i])
#     # plt.xticks(x, files, rotation=45, ha='right')
#     # plt.ylabel("Czas działania [s]")
#     # plt.title("Porównanie czasu działania")
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.grid(axis='y')
#     # plt.savefig("czasy_algorytmy.png")
#     # plt.show()
#
#     # Print some basic info
#     # print("Depot:", cvrp_graph.get_depot())
#     # print("Vehicle Capacity:", cvrp_graph.get_capacity())
#     # print("Demands:", cvrp_graph.get_demands())
#     # print("Graph Nodes:", cvrp_graph.get_graph().nodes(data=True))
#     # print("Graph Edges:", list(cvrp_graph.get_graph().edges(data=True)))
#
#     results = {}
#
#     # Uruchomienie algorytmu zachłannego (Nearest Neighbor)
#     # start = time.time()
#     # solver_greedy = CVRPNearestNeighbor(cvrp_graph)
#     # routes_greedy, total_cost_greedy = solver_greedy.solve()
#     # end = time.time()
#     # results['Greedy'] = {'cost': total_cost_greedy, 'time': end - start}
#     #
#     # # Algorytm mrówkowy
#     # start = time.time()
#     # ant_solver = AntColonyCVRP(cvrp_graph)
#     # routes_aco, total_cost_aco = ant_solver.solve()
#     # end = time.time()
#     # results['ACO_basic'] = {'cost': total_cost_aco, 'time': end - start}
#     #
#     # start = time.time()
#     # ant_solver = AntColonyCVRP(cvrp_graph)
#     # routes_aco, total_cost_aco = ant_solver.solve_elite_ants()
#     # end = time.time()
#     # results['ACO_elite'] = {'cost': total_cost_aco, 'time': end - start}
#     #
#     # start = time.time()
#     # ant_solver = AntColonyCVRP(cvrp_graph)
#     # routes_aco, total_cost_aco = ant_solver.solve_2_opt()
#     # end = time.time()
#     # results['ACO_2opt'] = {'cost': total_cost_aco, 'time': end - start}
#     #
#     # # save_results(routes_greedy, total_cost_greedy, routes_aco, total_cost_aco)
#     #
#     # print("Greedy solution:", routes_greedy, "Cost:", total_cost_greedy)
#     # print("ACO solution:", routes_aco, "Cost:", total_cost_aco)
#     #
#     # # Dane
#     # labels = list(results.keys())
#     # costs = [results[k]['cost'] for k in labels]
#     # times = [results[k]['time'] for k in labels]
#     #
#     # # 1. Koszt (jakość rozwiązania)
#     # plt.figure(figsize=(10, 5))
#     # plt.bar(labels, costs, color='skyblue')
#     # plt.title('Porównanie jakości (koszt całkowity)')
#     # plt.ylabel('Całkowity koszt trasy')
#     # plt.xlabel('Algorytm')
#     # plt.grid(True, axis='y')
#     # plt.tight_layout()
#     # plt.savefig('../results/cost_comparison.png')
#     # plt.show()
#     #
#     # # 2. Czas działania
#     # plt.figure(figsize=(10, 5))
#     # plt.bar(labels, times, color='salmon')
#     # plt.title('Porównanie czasu działania')
#     # plt.ylabel('Czas [s]')
#     # plt.xlabel('Algorytm')
#     # plt.grid(True, axis='y')
#     # plt.tight_layout()
#     # plt.savefig('../results/time_comparison.png')
#     # plt.show()
#     # Ładowanie danych
#     # graph = CVRPGraph()
#     # graph.load_vrp_data("../data/A-n32-k5.vrp", 5, 300)
#     #
#     # # Funkcja do testowania parametrów
#     # def test_param_effect(param_name, values, fixed_params):
#     #     results = []
#     #
#     #     for value in values:
#     #         kwargs = fixed_params.copy()
#     #         kwargs[param_name] = value
#     #
#     #         aco = AntColonyCVRP(graph, **kwargs)
#     #         best_solution, cost = aco.solve()
#     #         results.append(cost)
#     #         print(f"{param_name}={value} => cost={cost}")
#     #
#     #     return results
#     #
#     # # Zakresy parametrów
#     # alpha_vals = [0.5, 1, 1.5, 2, 3]
#     # beta_vals = [2, 3, 4, 5, 6]
#     # rho_vals = [0.3, 0.5, 0.6, 0.8, 0.9]
#     # q_vals = [10, 50, 100, 200, 300]
#     #
#     # # Ustalone wartości domyślne
#     # default_params = {
#     #     "num_iterations": 100,
#     #     "num_ants": 100,
#     #     "alpha": 1,
#     #     "beta": 5,
#     #     "evaporation_rate": 0.6,
#     #     "q": 100,
#     #     "seed": 0
#     # }
#     #
#     # # Testowanie i zbieranie wyników
#     # alpha_results = test_param_effect("alpha", alpha_vals, default_params)
#     # beta_results = test_param_effect("beta", beta_vals, default_params)
#     # rho_results = test_param_effect("evaporation_rate", rho_vals, default_params)
#     # q_results = test_param_effect("q", q_vals, default_params)
#     #
#     # # Wykresy
#     # plt.figure(figsize=(12, 8))
#     #
#     # plt.subplot(2, 2, 1)
#     # plt.plot(alpha_vals, alpha_results, marker='o')
#     # plt.title("Wpływ α (feromonów)")
#     # plt.xlabel("α")
#     # plt.ylabel("Koszt trasy")
#     #
#     # plt.subplot(2, 2, 2)
#     # plt.plot(beta_vals, beta_results, marker='o', color='green')
#     # plt.title("Wpływ β (heurystyki)")
#     # plt.xlabel("β")
#     # plt.ylabel("Koszt trasy")
#     #
#     # plt.subplot(2, 2, 3)
#     # plt.plot(rho_vals, rho_results, marker='o', color='orange')
#     # plt.title("Wpływ współczynnika parowania (ρ)")
#     # plt.xlabel("ρ")
#     # plt.ylabel("Koszt trasy")
#     #
#     # plt.subplot(2, 2, 4)
#     # plt.plot(q_vals, q_results, marker='o', color='red')
#     # plt.title("Wpływ Q (feromon)")
#     # plt.xlabel("Q")
#     # plt.ylabel("Koszt trasy")
#     #
#     # plt.tight_layout()
#     # plt.show()
#
#
# if __name__ == "__main__":
#     main()