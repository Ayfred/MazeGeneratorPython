import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

""" 
    This code was inspired by the following source: 
    https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96
    The code was modified (creation of class, methods, and attributes) to fit the project's requirements
     
    This class generates a maze using the recursive backtracking algorithm
    The maze is represented as a 2D numpy array where 1 represents a wall and 0 represents a cell
    The entrance and exit are represented as the first and last cell of the maze respectively
    The class also contains a method to draw the maze using matplotlib
    The class also contains a method to animate the solution path using matplotlib 
    
    @Author: Maxime Mu
    @Date: 2024-02-18
"""


class GenerateMaze:
    entrance = (1, 0)

    def __init__(self, dim):
        self.height = dim
        self.width = dim
        self.maze = np.ones((self.width * 2 + 1, self.height * 2 + 1))
        self.wall = 1
        self.cell = 0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.entrance = (1, 0)
        self.exit = (self.maze.shape[0] - 2, self.maze.shape[1] - 1)

        self.generateMaze()

    def generateMaze(self):
        # Define the starting point
        x, y = (0, 0)
        self.maze[x][y] = self.cell

        # Initialize the stack with the starting point
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]

            directions = self.directions
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self.width > nx >= 0 <= ny < self.height and self.maze[2 * nx + 1][2 * ny + 1] == self.wall:
                    self.maze[2 * nx + 1][2 * ny + 1] = self.cell
                    self.maze[2 * x + 1 + dx][2 * y + 1 + dy] = self.cell
                    stack.append((nx, ny))
                    break

            else:
                stack.pop()

        # Create entrance and exit
        self.maze[self.entrance] = self.cell
        self.maze[self.exit] = self.cell

    def drawMaze(self, path=None, iterations=0):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set the border color to white
        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')

        # Draw the solution path if it exists
        if path is not None:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            ax.plot(x_coords, y_coords, color='red', linewidth=2)

        ax.set_xticks([])
        ax.set_yticks([])

        iterations_text = ax.text(0.5, 1.05, f"Total Iterations: {iterations}", ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Solution Path Animation", fontsize=16)
        iterations_text.set_text(f"Total Iterations: {iterations}")

        # Draw entry and exit arrows
        ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
        ax.arrow(self.maze.shape[1] - 1, self.maze.shape[0] - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3,
                 head_length=0.3)

        plt.show()

    def animate_path(self, path, iterations):
        fig, ax = plt.subplots(figsize=(10, 10))

        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')

        if path is not None:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]

            for i in range(len(x_coords)):
                ax.plot(x_coords[:i + 1], y_coords[:i + 1], color='red', linewidth=2)
                plt.pause(0.0000000000000000001)

        ax.set_xticks([])
        ax.set_yticks([])

        iterations_text = ax.text(0.5, 1.05, f"Total Iterations: {iterations}", ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Solution Path Animation", fontsize=16)
        iterations_text.set_text(f"Total Iterations: {iterations}")

        # Draw entry and exit arrows
        ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
        ax.arrow(self.maze.shape[1] - 1, self.maze.shape[0] - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3,
                 head_length=0.3)

        plt.show()







    def value_iteration_animation(self, path, values, iterations):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')
        ax.set_title("Value Iteration Visualization", fontsize=16)

        # Normalize the values matrix to range from 0 to 1 for color intensity
        min_val, max_val = np.min(values), np.max(values)
        normalized_values = (values - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(values)

        # Colormap for visualizing the values
        cmap = plt.cm.inferno

        plotted_path = []  # List to store the plotted path coordinates

        def draw_grid():
            ax.cla()
            ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')
            ax.set_title("Value Iteration Visualization", fontsize=16)
            for i in range(len(normalized_values)):
                for j in range(len(normalized_values[i])):
                    if self.maze[i, j] != 1:  # Assuming 1 represents a wall
                        val_color = cmap(normalized_values[i, j])
                        ax.text(j, i, f"{normalized_values[i, j]:.2f}", ha='center', va='center', fontsize=10, color=val_color)

        def animate(frame):
            draw_grid()
            if path is not None and frame < len(path):
                y, x = path[frame]
                plotted_path.append((x, y))
                ax.plot(*zip(*plotted_path), color='red', linewidth=2)

            ax.set_xticks([])
            ax.set_yticks([])

            # Display the total number of iterations
            ax.text(0.5, 1.05, f"Total Iterations: {iterations}", ha='center', va='center', transform=ax.transAxes, fontsize=14)

            if frame == len(path) - 1:
                ani.event_source.stop()

        ani = animation.FuncAnimation(fig, animate, frames=len(path), interval=100, repeat=False)
        plt.show()
        return fig, ax, ani






        
    def policy_iteration_animation(self, path, policy, values, iterations):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')
        ax.set_title("Policy Iteration Visualization", fontsize=14)

        # Normalize the values for coloring
        norm = Normalize(values.min(), values.max())
        cmap = plt.cm.viridis

        # Function to draw an arrow with color intensity based on value
        def draw_arrow(i, j, color='blue'):
            arrow_size = 0.5
            arrow_head_width = 0.3
            arrow_head_length = 0.3

            # Directional arrows
            if policy[i, j] == 3:  # Up
                ax.arrow(j, i, 0, -arrow_size, head_width=arrow_head_width, head_length=arrow_head_length, fc=color, ec=color)
            elif policy[i, j] == 0:  # Right
                ax.arrow(j, i, arrow_size, 0, head_width=arrow_head_width, head_length=arrow_head_length, fc=color, ec=color)
            elif policy[i, j] == 1:  # Down
                ax.arrow(j, i, 0, arrow_size, head_width=arrow_head_width, head_length=arrow_head_length, fc=color, ec=color)
            elif policy[i, j] == 2:  # Left
                ax.arrow(j, i, -arrow_size, 0, head_width=arrow_head_width, head_length=arrow_head_length, fc=color, ec=color)

        # Draw all arrows with color intensity based on values
        for i in range(len(policy)):
            for j in range(len(policy[i])):
                if self.maze[i, j] != 1:  # Assuming 1 represents a wall
                    val_color = cmap(norm(values[i, j]))
                    draw_arrow(i, j, color=val_color)

        # Create a text object for iteration count
        iteration_text = ax.text(0.5, 1.05, "", ha='center', va='center', transform=ax.transAxes, fontsize=14)

        def animate(frame):
            if frame < len(path):
                y, x = path[frame]
                val_color = cmap(norm(values[y, x]))
                draw_arrow(y, x, color=val_color)

                # Update the red path
                x_coords = [p[1] for p in path[:frame + 1]]
                y_coords = [p[0] for p in path[:frame + 1]]
                ax.plot(x_coords, y_coords, color='red', linewidth=2)

            ax.set_xticks([])
            ax.set_yticks([])
        
        iteration_text.set_text(f"Total Iterations: {iterations}")

        ani = animation.FuncAnimation(fig, animate, frames=iterations, interval=100, repeat=False)
        plt.show()
        return fig, ax, ani
