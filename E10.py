import numpy as np
from queue import PriorityQueue

class Node:
    def __init__(self, level, value, weight, bound, taken):
        self.level = level  # Depth in the decision tree
        self.value = value  # Total value collected so far
        self.weight = weight  # Total weight collected so far
        self.bound = bound  # Upper bound of the maximum value achievable from this node
        self.taken = taken  # Projects selected (binary: 1 or 0)

    def __lt__(self, other):
        # Priority Queue uses this for sorting. Higher bound nodes are processed first.
        return self.bound > other.bound


def calculate_bound(node, n, max_weight, values, weights):
    """Calculate the upper bound on the total value starting from the given node."""
    if node.weight >= max_weight:
        return 0

    bound = node.value
    total_weight = node.weight
    level = node.level + 1

    # Add as much value as possible without exceeding max_weight
    while level < n and total_weight + weights[level] <= max_weight:
        total_weight += weights[level]
        bound += values[level]
        level += 1

    # Add fractional value for the next project if applicable
    if level < n:
        bound += (max_weight - total_weight) * (values[level] / weights[level])

    return bound


def branch_and_bound(values, weights, max_weight):
    """Solve the 0-1 knapsack problem using the branch and bound algorithm."""
    n = len(values)
    pq = PriorityQueue()

    # Initial node (root)
    root = Node(-1, 0, 0, 0.0, [])
    root.bound = calculate_bound(root, n, max_weight, values, weights)
    pq.put(root)

    max_value = 0
    best_combination = None

    while not pq.empty():
        current = pq.get()

        # Only explore nodes with promising bounds
        if current.bound > max_value:
            level = current.level + 1

            # Branch for taking the current project
            if level < n:
                taken_node = Node(
                    level,
                    current.value + values[level],
                    current.weight + weights[level],
                    0.0,
                    current.taken + [1],
                )

                if taken_node.weight <= max_weight and taken_node.value > max_value:
                    max_value = taken_node.value
                    best_combination = taken_node.taken

                taken_node.bound = calculate_bound(taken_node, n, max_weight, values, weights)

                if taken_node.bound > max_value:
                    pq.put(taken_node)

            # Branch for not taking the current project
            not_taken_node = Node(
                level,
                current.value,
                current.weight,
                0.0,
                current.taken + [0],
            )
            not_taken_node.bound = calculate_bound(not_taken_node, n, max_weight, values, weights)

            if not_taken_node.bound > max_value:
                pq.put(not_taken_node)

    return max_value, best_combination


if __name__ == "__main__":
    import pandas as pd

    # Example input data
    data = {
        "Project": [1, 2, 3, 4, 5, 6],
        "Revenue": [15, 20, 5, 25, 22, 17],
        "Days": [51, 60, 35, 60, 53, 10],
    }

    df = pd.DataFrame(data)
    values = df["Revenue"].to_numpy()
    weights = df["Days"].to_numpy()
    max_days = 150  # Example total time resource constraint

    # Solve the problem
    max_revenue, best_projects_binary = branch_and_bound(values, weights, max_days)

    # Map binary solution back to project numbers
    selected_projects = [data["Project"][i] for i, taken in enumerate(best_projects_binary) if taken == 1]

    print(f"Maximum Revenue: {max_revenue}")
    print(f"Projects to take: {selected_projects}")
