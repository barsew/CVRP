import numpy as np
import random
import os
import json

from torch.distributed.tensor import empty


class AntColonyCVRP:
    def __init__(self, graph, num_iterations=300, num_ants=10, alpha=1, beta=6, evaporation_rate=0.9, q=100, seed=0):
        """
        Algorytm mrówkowy dla CVRP.

        :param graph: Obiekt klasy CVRPGraph
        :param num_iterations: Liczba iteracji
        :param num_ants: Liczba mrówek w populacji
        :param alpha: Współczynnik wpływu feromonów
        :param beta: Współczynnik wpływu heurystyki (odległości)
        :param evaporation_rate: Współczynnik parowania feromonów
        :param q: Stała do aktualizacji feromonów
        """
        self.graph = graph
        self.num_vehicles = graph.num_of_trucks
        self.max_route_length = graph.max_dist
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.pheromone = self.initialize_pheromones()
        self.seed = seed
        self.distance_matrix = self.compute_distance_matrix()

    def compute_distance_matrix(self):
        """Precomputuje macierz odległości między wszystkimi wierzchołkami."""
        nodes = list(self.graph.get_graph().nodes())
        n = len(nodes)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = self.graph.euclidean_distance(self.graph.get_coordinates(nodes[i]), self.graph.get_coordinates(nodes[j]))
        return dist_matrix

    def initialize_pheromones(self):
        """Inicjalizuje wartości feromonów na każdej krawędzi."""
        pheromone = {}
        for edge in self.graph.get_graph().edges():
            v, u = edge
            pheromone[(v, u)] = 1.0
            pheromone[(u, v)] = 1.0
        return pheromone

    def heuristic(self, node1, node2):
        """Oblicza heurystykę jako odwrotność odległości."""
        distance = self.distance_matrix[node1 - 1, node2 - 1]
        return 1 / distance if distance > 0 else 1

    def select_next_node(self, current_node, unvisited, remaining_capacity, remaining_distance):
        """Wybiera następnego klienta na podstawie feromonów i heurystyki."""
        probabilities = []
        nodes = []

        for node in unvisited:
            demand = self.graph.get_demands()[node]
            distance = self.graph.euclidean_distance(self.graph.get_coordinates(current_node), self.graph.get_coordinates(node))

            if demand <= remaining_capacity and distance <= remaining_distance: # powinno sie patrzec czy na powort do depot starczy
                tau = self.pheromone.get((current_node, node), 1.0) ** self.alpha
                eta = self.heuristic(current_node, node) ** self.beta
                probabilities.append(tau * eta)
                nodes.append(node)

        if not nodes:
            return None  # Brak dostępnych klientów

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        #np.random.seed(self.seed)

        return np.random.choice(nodes, p=probabilities)

    def two_opt(self, route):
        """
        Wykonuje lokalną optymalizację trasy za pomocą heurystyki 2-opt.

        :param route: Lista odwiedzanych wierzchołków (w tym powrotu do depotu)
        :return: Optymalizowana trasa
        """
        subroutes = self.split_route_by_depot(route, 1)
        for subroute in subroutes:
            improved = True
            while improved:
                improved = False
                for i in range(1, len(subroute) - 2):
                    for j in range(i + 1, len(subroute) - 1):
                        if j - i == 1:
                            continue
                        new_subroute = subroute[:i] + subroute[i:j + 1][::-1] + subroute[j + 1:]
                        if self.calculate_route_cost(new_subroute) < self.calculate_route_cost(subroute):
                            subroute = new_subroute
                            improved = True
        return self.join_subroutes_with_depots(subroutes, 1)

    def split_route_by_depot(self, route, depot=1):
        subroutes = []
        current = []

        for node in route:
            if node == depot:
                if current:
                    subroutes.append(current)
                    current = []
            else:
                current.append(node)

        if current:
            subroutes.append(current)

        return subroutes

    def join_subroutes_with_depots(self, subroutes, depot=1):
        full_route = []
        for subroute in subroutes:
            full_route.append(depot)
            full_route.extend(subroute)
        full_route.append(depot)
        return full_route

    def calculate_route_cost(self, route):
        """Oblicza całkowity koszt trasy."""
        return sum(self.distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

    def construct_solution_2_opt(self):
        """Buduje rozwiązanie z wykorzystaniem mrówek i optymalizuje je heurystyką 2-opt."""
        depot = self.graph.get_depot()
        routes = [[] for _ in range(self.num_ants)]
        total_cost = []

        for ant in range(self.num_ants):
            ant_route = [depot]
            unvisited = set(self.graph.customers.keys())
            remaining_capacity = self.graph.get_capacity()
            remaining_distance = self.max_route_length
            current_node = depot
            current_visits_of_depot = 1

            while unvisited and current_visits_of_depot <= self.num_vehicles:
                next_node = self.select_next_node(current_node, unvisited, remaining_capacity, remaining_distance)
                if next_node is None:
                    current_visits_of_depot += 1
                    remaining_capacity = self.graph.get_capacity()
                    remaining_distance = self.max_route_length
                    ant_route.append(depot)
                    continue

                ant_route.append(next_node)
                remaining_capacity -= self.graph.get_demands()[next_node]
                remaining_distance -= self.graph.euclidean_distance(self.graph.get_coordinates(current_node),
                                                                    self.graph.get_coordinates(next_node))
                unvisited.remove(next_node)
                current_node = next_node

            if current_visits_of_depot > self.num_vehicles and len(unvisited) != 0:
                routes[ant] = []
                total_cost.append(float('inf'))
            else:
                ant_route.append(depot)
                optimized_route = self.two_opt(ant_route)
                routes[ant] = optimized_route
                total_cost.append(self.calculate_route_cost(optimized_route))

        return routes, total_cost

    def construct_solution(self):
        """Buduje rozwiązanie z wykorzystaniem mrówek."""
        depot = self.graph.get_depot()
        routes = [[] for _ in range(self.num_ants)]
        total_cost = []

        for ant in range(self.num_ants):

            ant_route = [depot]
            unvisited = set(self.graph.customers.keys())
            remaining_capacity = self.graph.get_capacity()
            remaining_distance = self.max_route_length
            current_node = depot
            current_visits_of_depot = 1    # === current truck

            while unvisited and current_visits_of_depot <= self.num_vehicles:
                next_node = self.select_next_node(current_node, unvisited, remaining_capacity, remaining_distance)

                if next_node is None:  # Jeśli nie ma więcej możliwych ruchów, ale sa nieodwiedzone to powrot do depotu i zmiana ciezarowki
                    current_visits_of_depot += 1
                    remaining_capacity = self.graph.get_capacity()
                    remaining_distance = self.max_route_length
                    ant_route.append(depot)
                    continue

                ant_route.append(next_node)
                remaining_capacity -= self.graph.get_demands()[next_node]
                remaining_distance -= self.graph.euclidean_distance(self.graph.get_coordinates(current_node), self.graph.get_coordinates(next_node))
                unvisited.remove(next_node)
                current_node = next_node

            if current_visits_of_depot > self.num_vehicles and len(unvisited) != 0:  # nie mamy trasy
                routes[ant] = []
                total_cost.append(float('inf'))
            else:
                ant_route.append(depot)
                routes[ant] = ant_route
                total_cost.append(self.calculate_route_cost(ant_route))

        return routes, total_cost

    def update_pheromones(self, solutions, cost):
        # Evaporacja feromonów
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)

        # Aktualizacja feromonów na podstawie tras mrówek
        for i, solution in enumerate(solutions):
            if cost[i] == float('inf'):
                continue  # Nie aktualizujemy, jeśli rozwiązanie jest złe

            pheromone_deposit = self.q / cost[i]

            for j in range(len(solution) - 1):
                e = (solution[j], solution[j + 1])
                reversed_e = (solution[j + 1], solution[j])

                self.pheromone[e] = self.pheromone.get(e, 1.0) + pheromone_deposit
                self.pheromone[reversed_e] = self.pheromone.get(reversed_e, 1.0) + pheromone_deposit

    def update_pheromones_elite_ants(self, solutions, cost, best_solution, best_cost, elite_weight=5):
        # Evaporacja feromonów
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)

        # Aktualizacja feromonów na podstawie rozwiązań mrówek
        for i, solution in enumerate(solutions):
            if cost[i] == float('inf'):
                continue  # Ignorujemy złe rozwiązania

            pheromone_deposit = self.q / cost[i]

            for j in range(len(solution) - 1):
                e = (solution[j], solution[j + 1])
                reversed_e = (solution[j + 1], solution[j])

                self.pheromone[e] = self.pheromone.get(e, 1.0) + pheromone_deposit
                self.pheromone[reversed_e] = self.pheromone.get(reversed_e, 1.0) + pheromone_deposit

        elite_pheromone_deposit = elite_weight * (self.q / best_cost)

        for j in range(len(best_solution) - 1):
            e = (best_solution[j], best_solution[j + 1])
            reversed_e = (best_solution[j + 1], best_solution[j])

            self.pheromone[e] = self.pheromone.get(e, 1.0) + elite_pheromone_deposit
            self.pheromone[reversed_e] = self.pheromone.get(reversed_e, 1.0) + elite_pheromone_deposit

    def calculate_route_cost(self, route):
        """Oblicza koszt trasy jako sumę długości krawędzi."""
        return sum(self.graph.euclidean_distance(self.graph.get_coordinates(route[i]), self.graph.get_coordinates(route[i+1])) for i in range(len(route) - 1))

    def solve(self):
        """Uruchamia algorytm mrówkowy."""
        best_rout = None
        best_cost = float('inf')

        for i in range(self.num_iterations):
            routes, cost = self.construct_solution()
            #routes, cost = self.construct_solution_2_opt()
            min_cost = min(cost)
            idx = cost.index(min_cost)

            if min_cost < best_cost:
                best_rout, best_cost = routes[idx], min(cost)
            self.update_pheromones(routes, cost)
            #self.update_pheromones_elite_ants(routes, cost, best_rout, best_cost)
            # if i % 50 == 0:
            #     print(f"Iteration {i}: Best Cost = {best_cost}")

        return best_rout, best_cost

    def solve_elite_ants(self):
        """Uruchamia algorytm mrówkowy."""
        best_rout = None
        best_cost = float('inf')

        for i in range(self.num_iterations):
            routes, cost = self.construct_solution()
            min_cost = min(cost)
            idx = cost.index(min_cost)

            if min_cost < best_cost:
                best_rout, best_cost = routes[idx], min(cost)
            self.update_pheromones_elite_ants(routes, cost, best_rout, best_cost)
            # if i % 50 == 0:
            #     print(f"Iteration {i}: Best Cost = {best_cost}")

        return best_rout, best_cost

    def solve_2_opt(self):
        best_rout = None
        best_cost = float('inf')

        for i in range(self.num_iterations):
            routes, cost = self.construct_solution_2_opt()
            min_cost = min(cost)
            idx = cost.index(min_cost)

            if min_cost < best_cost:
                best_rout, best_cost = routes[idx], min(cost)
            self.update_pheromones(routes, cost)
            # if i % 50 == 0:
            #     print(f"Iteration {i}: Best Cost = {best_cost}")

        return best_rout, best_cost