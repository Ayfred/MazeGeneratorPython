import sys
import algorithms

"""
    This is the main file of the project. It is used to run the algorithms and compare them.
    The user can choose the algorithm to run and the dimension of the maze.
    The user can also choose to animate the process or not.
    
    @Author: Maxime Mu
    @Date: 2024-02-18
"""


def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python Main.py <option> <animate>")
        print(
            "Invalid option. Please use '-1' for Bfs, '-2' for Dfs, '-3' for A*, '-4' for Value Iteration, "
            "'-5' for Policy Iteration, '-6' for Comparison.")
        print("Add 'y' at the end to animate the process.")
        print("Example: python Main.py -1 y")
        return

    option = sys.argv[1]

    if len(sys.argv) == 3 and sys.argv[2] == 'y':
        animate = True
    else:
        animate = False

    if option == '-1':
        dim = input("Enter the dimension of the maze: ")
        algorithms.Algorithms(int(dim), 1, animation=animate)
    elif option == '-2':
        dim = input("Enter the dimension of the maze: ")
        algorithms.Algorithms(int(dim), 2, animation=animate)
    elif option == '-3':
        dim = input("Enter the dimension of the maze: ")
        algorithms.Algorithms(int(dim), 3, animation=animate)
    elif option == '-4':
        dim = input("Enter the dimension of the maze: ")
        algorithms.Algorithms(int(dim), 4, animation=animate)
    elif option == '-5':
        dim = input("Enter the dimension of the maze: ")
        algorithms.Algorithms(int(dim), 5, animation=animate)
    elif option == '-6':
        print("This option is for comparison between all the 5 algorithms.")
        print("Please choose in order the dimension of the maze")
        dim = input("Enter the first dimension of the maze: ")
        dim2 = input("Enter the second dimension of the maze: ")
        dim3 = input("Enter the third dimension of the maze: ")
        algorithms.Algorithms(int(dim), 6, int(dim2), int(dim3))
    else:
        print(
            "Invalid option. Please use '-1' for Bfs, '-2' for Dfs, '-3' for A*, '-4' for Value Iteration, "
            "'-5' for Policy Iteration, '-6' for Comparison.")
        print("Add 'y' at the end to animate the process.")
        print("Example: python Main.py -1 y")
    return


if __name__ == '__main__':
    main()
