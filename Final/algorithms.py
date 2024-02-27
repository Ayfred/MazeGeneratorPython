import generate_maze
import numpy as np
from queue import Queue
import heapq
import time
import pandas as pd
import matplotlib.pyplot as plt

"""
    This class contains the algorithms to solve the maze
    The algorithms are: BFS, DFS, A*, Value Iteration, Policy Iteration
    
    @Author: Maxime Mu
    @Date: 2024-02-18
"""


class Algorithms:
    def __init__(self, dim, option, dim2=None, dim3=None, animation=False):
        self.mazeGenerator = generate_maze.GenerateMaze(dim)
        self.path = None
        self.iterations = 0
        self.option(option, dim, dim2, dim3, animation)

    def option(self, option, dim, dim2, dim3, animation):
        if option == 1:
            print("Starting solving the maze using BFS algorithm")
            self.path, self.iterations = self.bfs_algorithm()
            print("Drawing the maze with the solution path")

            if animation:
                self.mazeGenerator.animate_path(self.path, self.iterations)
            else:
                self.mazeGenerator.drawMaze(self.path, self.iterations)

        elif option == 2:
            print("Starting solving the maze using DFS algorithm")
            self.path, self.iterations = self.dfs_algorithm()
            print("Drawing the maze with the solution path")

            if animation:
                self.mazeGenerator.animate_path(self.path, self.iterations)
            else:
                self.mazeGenerator.drawMaze(self.path, self.iterations)

        elif option == 3:
            print("Starting solving the maze using A* algorithm")
            self.path, self.iterations = self.astar_algorithm()
            print("Drawing the maze with the solution path")

            if animation:
                self.mazeGenerator.animate_path(self.path, self.iterations)
            else:
                self.mazeGenerator.drawMaze(self.path, self.iterations)

        elif option == 4:
            print("Starting solving the maze using Value Iteration algorithm")
            self.path, self.iterations, values = self.value_iteration(
                reward=create_reward(self.mazeGenerator.maze, self.mazeGenerator.exit))
            print("Drawing the maze with the solution path")

            if animation:
                self.mazeGenerator.value_iteration_animation(self.path, values, self.iterations)
            else:
                self.mazeGenerator.drawMaze(self.path, self.iterations)

        elif option == 5:
            print("Starting solving the maze using Policy Iteration algorithm")
            self.path, self.iterations, policy, values = self.policy_iteration(
                create_reward(self.mazeGenerator.maze, self.mazeGenerator.exit))
            print("Drawing the maze with the solution path")

            if animation:
                self.mazeGenerator.policy_iteration_animation(self.path, policy, values, self.iterations)
            else: 
                self.mazeGenerator.drawMaze(self.path, self.iterations)

        elif option == 6:
            results_time = []
            results_iterations = []
            dims = [dim, dim2, dim3]
            for i in range(len(dims)):
                print("Processing maze dimension: ", dims[i])
                self.mazeGenerator = generate_maze.GenerateMaze(dims[i])

                # Average time and iterations
                average_time_bfs = 0
                average_time_dfs = 0
                average_time_astar = 0
                average_time_value_iteration = 0
                average_time_policy_iteration = 0

                average_iterations_bfs = 0
                average_iterations_dfs = 0
                average_iterations_astar = 0
                average_iterations_value_iteration = 0
                average_iterations_policy_iteration = 0

                repetitions = 50 

                for j in range(repetitions):
                    self.mazeGenerator = generate_maze.GenerateMaze(dims[i])
                    start = time.time()
                    self.path, bfs_iterations = self.bfs_algorithm()
                    end = time.time()
                    average_time_bfs += (end - start)
                    average_iterations_bfs += bfs_iterations
                average_time_bfs /= repetitions
                average_iterations_bfs /= repetitions

                for j in range(repetitions):
                    self.mazeGenerator = generate_maze.GenerateMaze(dims[i])
                    start = time.time()
                    self.path, dfs_iterations = self.dfs_algorithm()
                    end = time.time()
                    average_time_dfs += (end - start)
                    average_iterations_dfs += dfs_iterations
                average_time_dfs /= repetitions
                average_iterations_dfs /= repetitions
                
                for j in range(repetitions):
                    self.mazeGenerator = generate_maze.GenerateMaze(dims[i])
                    start = time.time()
                    self.path, astar_iterations = self.astar_algorithm()
                    end = time.time()
                    average_time_astar += (end - start)
                    average_iterations_astar += astar_iterations
                average_time_astar /= repetitions
                average_iterations_astar /= repetitions

                for j in range(repetitions):
                    self.mazeGenerator = generate_maze.GenerateMaze(dims[i])
                    start = time.time()
                    self.path, value_iterations, values = self.value_iteration(
                        reward=create_reward(self.mazeGenerator.maze, self.mazeGenerator.exit))
                    end = time.time()
                    average_time_value_iteration += (end - start)
                    average_iterations_value_iteration += value_iterations
                average_time_value_iteration /= repetitions
                average_iterations_value_iteration /= repetitions

                for j in range(repetitions):
                    self.mazeGenerator = generate_maze.GenerateMaze(dims[i])
                    start = time.time()
                    self.path, policy_iterations, policy, values = self.policy_iteration(
                        create_reward(self.mazeGenerator.maze, self.mazeGenerator.exit))
                    end = time.time()
                    average_time_policy_iteration += (end - start)
                    average_iterations_policy_iteration += policy_iterations
                average_time_policy_iteration /= repetitions
                average_iterations_policy_iteration /= repetitions

                results_time.append([dims[i], average_time_bfs, average_time_dfs, average_time_astar, average_time_value_iteration, average_time_policy_iteration])
                results_iterations.append([dims[i], average_iterations_bfs, average_iterations_dfs, average_iterations_astar, average_iterations_value_iteration, average_iterations_policy_iteration])

            columns = ['Dimension', 'BFS', 'DFS', 'A*', 'Value Iteration', 'Policy Iteration']
            df = pd.DataFrame(results_time, columns=columns)
            # print(df)
            df.plot(x='Dimension', y=['BFS', 'DFS', 'A*', 'Value Iteration', 'Policy Iteration'], kind='line')
            plt.title('Time Comparison of Algorithms')
            plt.ylabel('Time (s)')
            plt.show()

            df_iterations = pd.DataFrame(results_iterations, columns=columns)
            # print(df_iterations)
            df_iterations.plot(x='Dimension', y=['BFS', 'DFS', 'A*', 'Value Iteration', 'Policy Iteration'],
                               kind='line')
            plt.title('Iterations Comparison of Algorithms')
            plt.ylabel('Number of Iterations')
            plt.show()
        else:
            print(
                "Invalid option. Please use '-1' for Bfs, '-2' for Dfs, '-3' for A*, '-4' for Value Iteration, "
                "'-5' for Policy Iteration, '-6' for Comparison.")

    def bfs_algorithm(self):
        start = self.mazeGenerator.entrance
        end = self.mazeGenerator.exit
        maze = self.mazeGenerator.maze
        possible_moves = self.mazeGenerator.directions

        q = Queue()
        q.put([start])

        iterations = 0
        while q.not_empty:
            path = q.get()
            current_pos = path[-1]

            if current_pos == end:
                return path, iterations
            else:
                for move in possible_moves:
                    new_position = (current_pos[0] + move[0], current_pos[1] + move[1])
                    if maze[new_position[0], new_position[1]] == self.mazeGenerator.cell:
                        if new_position not in path:
                            new_path = list(path)
                            new_path.append(new_position)
                            q.put(new_path)
            iterations += 1
        return None, iterations

    def dfs_algorithm(self):
        start = self.mazeGenerator.entrance
        end = self.mazeGenerator.exit
        maze = self.mazeGenerator.maze
        possible_moves = self.mazeGenerator.directions

        stack = [[start]]

        iterations = 0
        while stack:
            path = stack.pop()
            current_pos = path[-1]

            if current_pos == end:
                return path, iterations
            else:
                for move in possible_moves:
                    new_position = (current_pos[0] + move[0], current_pos[1] + move[1])
                    if maze[new_position[0], new_position[1]] == self.mazeGenerator.cell:
                        if new_position not in path:
                            new_track = list(path)
                            new_track.append(new_position)
                            stack.append(new_track)
            iterations += 1
        return None, iterations
    

    def create_path(self, came_from):
        current_pos = self.mazeGenerator.exit
        path = [current_pos]
        while current_pos != self.mazeGenerator.entrance:
            current_pos = came_from[current_pos]
            path.append(current_pos)
        return path[::-1]


    def astar_algorithm(self):
        start = self.mazeGenerator.entrance
        end = self.mazeGenerator.exit
        maze = self.mazeGenerator.maze
        possible_moves = self.mazeGenerator.directions

        open_set = []
        closed_set = set()
        came_from = {}

        gscore = {start: 0}
        fscore = {start: heuristic(start, end)}
        heapq.heappush(open_set, (0, start))

        iterations = 0
        while open_set:
            current_pos = heapq.heappop(open_set)[1]

            closed_set.add(current_pos)

            if current_pos == end:
                return self.create_path(came_from), iterations

            for move in possible_moves:
                new_position = (current_pos[0] + move[0], current_pos[1] + move[1])
                if maze[new_position[0], new_position[1]] == self.mazeGenerator.wall or new_position in closed_set:
                    continue

                tentative_gscore = gscore[current_pos] + 1
                if new_position not in [i[1] for i in open_set] or tentative_gscore < gscore[new_position]:
                    came_from[new_position] = current_pos
                    gscore[new_position] = tentative_gscore
                    fscore[new_position] = tentative_gscore + heuristic(new_position, end)
                    heapq.heappush(open_set, (fscore[new_position], new_position))
                    closed_set.add(new_position)
            iterations += 1
        return None, iterations


    def find_path(self, value, policy=None):
        path = [self.mazeGenerator.entrance]
        current_pos = self.mazeGenerator.entrance
        while current_pos != self.mazeGenerator.exit:

            if policy is None:
                next_position = self.next_position(current_pos, value)
            else:
                action = policy[current_pos[0], current_pos[1]]
                next_position = (current_pos[0] + self.mazeGenerator.directions[action][0],
                                 current_pos[1] + self.mazeGenerator.directions[action][1])

            if next_position is not None and is_within_bounds(self.mazeGenerator.maze, next_position):
                path.append(next_position)
                current_pos = next_position
            else:
                break
        return path 

    def next_position(self, current, value):
        max_value = float('-inf')
        next_position = None
        for move in self.mazeGenerator.directions:
            new_position = (current[0] + move[0], current[1] + move[1])
            if is_within_bounds(self.mazeGenerator.maze, new_position):
                if value[new_position] > max_value:
                    max_value = value[new_position]
                    next_position = new_position
        return next_position


    def value_iteration(self, reward, gamma=0.99, convergence_threshold=0.001):
        start = self.mazeGenerator.entrance
        end = self.mazeGenerator.exit
        maze = self.mazeGenerator.maze
        possible_moves = self.mazeGenerator.directions

        value = np.zeros(maze.shape)

        convergence = False
        iterations = 0
        while not convergence:
            delta = 0
            for i in range(maze.shape[0]):
                for j in range(maze.shape[1]):
                    if maze[i, j] == self.mazeGenerator.wall:  # Assuming wall cells are not to be updated
                        continue
                    temp = value[i, j]
                    max_value = float("-inf")
                    for move in possible_moves:
                        new_position = (i + move[0], j + move[1])
                        if is_within_bounds(maze, new_position):
                            # Assuming reward is a function or matrix that gives reward for moving to the new position
                            max_value = max(max_value, reward[new_position] + gamma * value[new_position])
                            #max_value = max(max_value, reward[i, j] + gamma * value[new_position])
                    value[i, j] = max_value  # Update the value to the maximum value found
                    delta = max(delta, abs(temp - value[i, j]))
            if delta < convergence_threshold:
                convergence = True
            iterations += 1

        path = self.find_path(value)

        return path, iterations, value

    def policy_iteration(self, reward, gamma=0.99, convergence_threshold=0.001):
        start = self.mazeGenerator.entrance
        end = self.mazeGenerator.exit
        maze = self.mazeGenerator.maze
        possible_moves = self.mazeGenerator.directions

        value = np.zeros(maze.shape)
        policy = np.random.randint(0, len(possible_moves), maze.shape)

        convergence = False
        iterations = 0
        while not convergence:
            while True:
                delta = 0
                for i in range(maze.shape[0]):
                    for j in range(maze.shape[1]):
                        if maze[i, j] == self.mazeGenerator.wall:
                            continue
                        temp = value[i, j]
                        action = policy[i, j]
                        sum_value = 0
                        new_position = (i + possible_moves[action][0], j + possible_moves[action][1])
                        if is_within_bounds(maze, new_position):
                            sum_value += reward[i, j] + gamma * value[new_position]
                        value[i, j] = sum_value
                        delta = max(delta, abs(temp - value[i, j]))
                if delta < convergence_threshold:
                    break

            policy_stable = True
            for i in range(maze.shape[0]):
                for j in range(maze.shape[1]):
                    if maze[i, j] == self.mazeGenerator.wall:
                        continue
                    old_action = policy[i, j]
                    max_value = float('-inf')
                    for k in range(len(possible_moves)):
                        new_position = (i + possible_moves[k][0], j + possible_moves[k][1])
                        if is_within_bounds(maze, new_position):
                            if value[new_position] > max_value:
                                max_value = value[new_position]
                                policy[i, j] = k
                    if old_action != policy[i, j]:
                        policy_stable = False
            if policy_stable:
                break
            iterations += 1

        path = self.find_path(value, policy)

        return path, iterations, policy, value

def is_within_bounds(maze, position):
    return 0 <= position[0] < maze.shape[0] and 0 <= position[1] < maze.shape[1]


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance





def create_reward(maze, end):
    reward = np.zeros(maze.shape)
    reward[end] = 300
    return reward