import numpy as np
import random
import os
import json


class AntColonyCVRP:
    def __init__(self, graph, num_iterations=300, num_ants=10, alpha=1, beta=5, evaporation_rate=0.6, q=100, seed=0):
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

    def initialize_pheromones(self):
        """Inicjalizuje wartości feromonów na każdej krawędzi."""
        pheromone = {}
        for edge in self.graph.get_graph().edges():
            pheromone[edge] = 1.0  # Początkowa ilość feromonów
        return pheromone

    def heuristic(self, node1, node2):
        """Oblicza heurystykę jako odwrotność odległości."""
        distance = self.graph.euclidean_distance(self.graph.get_coordinates(node1), self.graph.get_coordinates(node2))
        return 1 / distance if distance > 0 else 1

    def select_next_node(self, current_node, unvisited, remaining_capacity, remaining_distance):
        """Wybiera następnego klienta na podstawie feromonów i heurystyki."""
        probabilities = []
        nodes = []

        for node in unvisited:
            demand = self.graph.get_demands()[node]
            distance = self.graph.euclidean_distance(self.graph.get_coordinates(current_node), self.graph.get_coordinates(node))

            if demand <= remaining_capacity and distance <= remaining_distance:
                tau = self.pheromone.get((current_node, node), 1.0) ** self.alpha
                eta = self.heuristic(current_node, node) ** self.beta
                probabilities.append(tau * eta)
                nodes.append(node)

        if not nodes:
            return None  # Brak dostępnych klientów

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        np.random.seed(self.seed)

        return np.random.choice(nodes, p=probabilities)

    def construct_solution(self):
        """Buduje rozwiązanie z wykorzystaniem mrówek."""
        depot = self.graph.get_depot()
        routes = [[] for _ in range(self.num_vehicles)]
        total_cost = 0
        unvisited = set(self.graph.customers.keys())

        for vehicle in range(self.num_vehicles):
            if not unvisited:
                break

            vehicle_route = [depot]
            remaining_capacity = self.graph.get_capacity()
            remaining_distance = self.max_route_length
            current_node = depot

            while unvisited:
                next_node = self.select_next_node(current_node, unvisited, remaining_capacity, remaining_distance)

                if next_node is None:  # Jeśli nie ma więcej możliwych ruchów, zakończ trasę
                    break

                vehicle_route.append(next_node)
                remaining_capacity -= self.graph.get_demands()[next_node]
                remaining_distance -= self.graph.euclidean_distance(self.graph.get_coordinates(current_node), self.graph.get_coordinates(next_node))
                unvisited.remove(next_node)
                current_node = next_node

            vehicle_route.append(depot)
            routes[vehicle] = vehicle_route
            total_cost += self.calculate_route_cost(vehicle_route)

        return routes, total_cost

    def update_pheromones(self, solutions, cost):
        # Evaporacja feromonów
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)

        # Aktualizacja feromonów na podstawie rozwiązań mrówek
        for solution in solutions:
            pheromone_deposit = 1.0 / cost  # Możesz dostosować sposób depozytu

            for i in range(len(solution) - 1):
                edge = (solution[i], solution[i + 1])
                reversed_edge = (solution[i + 1], solution[i])  # Obsługa grafu nieskierowanego

                if edge not in self.pheromone:
                    self.pheromone[edge] = 0  # Inicjalizacja

                if reversed_edge not in self.pheromone:
                    self.pheromone[reversed_edge] = 0  # Inicjalizacja

                self.pheromone[edge] += pheromone_deposit
                self.pheromone[reversed_edge] += pheromone_deposit  # Jeśli graf jest nieskierowany

    def calculate_route_cost(self, route):
        """Oblicza koszt trasy jako sumę długości krawędzi."""
        return sum(self.graph.euclidean_distance(self.graph.get_coordinates(route[i]), self.graph.get_coordinates(route[i+1])) for i in range(len(route) - 1))

    def solve(self):
        """Uruchamia algorytm mrówkowy."""
        best_routes = None
        best_cost = float('inf')

        for _ in range(self.num_iterations):
            routes, cost = self.construct_solution()
            if cost < best_cost:
                best_routes, best_cost = routes, cost
            self.update_pheromones(routes, cost)

        return best_routes, best_cost
