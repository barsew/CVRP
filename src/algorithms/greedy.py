from src.utils.CVRPGraph import CVRPGraph
import json

class CVRPNearestNeighbor:
    def __init__(self, graph: CVRPGraph):
        self.graph = graph
        self.routes = []

    def find_nearest_neighbor(self, current_location, unvisited, remaining_capacity):
        """Znajduje najbliższego klienta, który mieści się w pojemności pojazdu."""
        nearest_customer = None
        min_distance = float('inf')

        for customer in unvisited:
            current_coords = self.graph.get_coordinates(current_location)  # Pobierz współrzędne depotu/klienta
            customer_coords = self.graph.customers[customer]  # Pobierz współrzędne klienta

            distance = self.graph.euclidean_distance(current_coords, customer_coords)
            if distance < min_distance and self.graph.demands[customer] <= remaining_capacity:
                nearest_customer = customer
                min_distance = distance

        return nearest_customer

    def solve(self):
        """Rozwiązuje problem CVRP metodą zachłanną."""
        unvisited = set(self.graph.customers.keys())
        total_cost = 0

        while unvisited:
            route = [1]  # Start at depot
            remaining_capacity = self.graph.capacity
            current_location = self.graph.depot
            current_dist = 0

            while True:
                nearest_customer = self.find_nearest_neighbor(current_location, unvisited, remaining_capacity)

                if nearest_customer is None:
                    # Jeśli nie możemy dodać więcej klientów, wracamy do depozytu
                    break

                current_dist += self.graph.euclidean_distance(self.graph.get_coordinates(current_location), self.graph.get_coordinates(nearest_customer))

                if (self.graph.euclidean_distance(self.graph.get_coordinates(nearest_customer), self.graph.get_coordinates(self.graph.get_depot())) + current_dist > self.graph.max_dist
                        or current_dist > self.graph.max_dist):
                    break

                route.append(nearest_customer)

                total_cost += self.graph.euclidean_distance(
                    self.graph.get_coordinates(current_location),
                    self.graph.get_coordinates(nearest_customer)
                )

                remaining_capacity -= self.graph.demands[nearest_customer]
                current_location = nearest_customer

                unvisited.remove(nearest_customer)

            route.append(1)  # Powrót do depotu
            total_cost += self.graph.euclidean_distance(
                self.graph.get_coordinates(current_location),
                self.graph.get_coordinates(self.graph.get_depot())
            )
            self.routes.append(route)
        print("Total cost of the solution:", total_cost)
        return self.routes, total_cost
        # self.save_results(self.routes, total_cost)

    def display_routes(self):
        """Wyświetla wygenerowane trasy."""
        for idx, route in enumerate(self.routes, start=1):
            print(f"Vehicle {idx}: {' -> '.join(map(str, route))}")

    # def save_results(self, routes, total_cost, filename="results"):
    #     """
    #     Zapisuje wyniki algorytmu do plików JSON i TXT.
    #
    #     :param routes: Lista tras dla pojazdów
    #     :param total_cost: Całkowity koszt rozwiązania
    #     :param filename: Nazwa pliku wyjściowego (bez rozszerzenia)
    #     """
    #     output_data = {
    #         "total_cost": total_cost,
    #         "routes": routes
    #     }
    #
    #     # Zapis do JSON
    #     json_filename = f"../results/{filename}.json"
    #     with open(json_filename, "w") as json_file:
    #         json.dump(output_data, json_file, indent=4)
    #
    #     # Zapis do TXT
    #     txt_filename = f"../results/{filename}.txt"
    #     with open(txt_filename, "w") as txt_file:
    #         txt_file.write(f"Total Cost: {total_cost}\n\n")
    #         for i, route in enumerate(routes):
    #             txt_file.write(f"Vehicle {i + 1}: {' -> '.join(map(str, route))}\n")
    #
    #     print(f"Results saved to: {json_filename} and {txt_filename}")
