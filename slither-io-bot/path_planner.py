import random, math

import numpy as np
from numpy.linalg import norm

d_influence = 100

class PathPlanner:
    def __init__(self, mode, snake_head) -> None:
        self.mode = mode
        self.snake_head = snake_head

    def get_radian(self, lines, pellets):
        if self.mode == 'apf':
            return self.artificial_potential_field_planner(lines, pellets)
        elif self.mode == 'cbf':
            return self.control_barrier_function_planner(lines, pellets)
        else:
            return self.get_random_direction()

    @staticmethod
    def get_random_direction():
        return random.uniform(0, math.pi * 2)

    def artificial_potential_field_planner(self, lines, pellets):
        attractive_potentials = self.calculate_attractive_potentials(pellets)
        total_potentials = attractive_potentials
        return np.arctan2(total_potentials[1], total_potentials[0])

    def calculate_attractive_potentials(self, pellets):
        grad_attractive_potentials = np.zeros((2, 1))
        counter = 0
        for pellet in pellets:
            (x, y) = pellet
            point = np.array(([x], [y]))
            grad_attractive_potentials += self.grad_attractive(self.snake_head, point)            
        return grad_attractive_potentials

    def calculate_repulsive_potentials(self, lines):
        grad_repulsive_potentials = np.zeros((2, 1))
        for i in range(lines.shape[0]):
            point1 = np.array([int(lines[i, 0])], [int(lines[i, 1])])
            point2 = np.array([int(lines[i, 2])], [int(lines[i, 3])])

    @staticmethod
    def get_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    def grad_attractive(self, point, goal):
        return 2 * (point - goal) * self.get_distance(point, goal)

    def grad_repulsive(self, point, obstacle_point):
        return self.get_distance(point, obstacle_point)

    def evaluate_repulsive_potential_at(x_eval, line):
        # calculates the closest point on the line
        # x_closest
        pass

    def control_barrier_function_planner(self, lines, pellets):
        pass


class Line:
    def __init__(self, point1, point2) -> None:
        self.point1 = point1
        self.point2 = point2

    def get_shortest_distance(self, x_eval):
        """
        Calculates the shortest distance from the line to a point
        """
        return np.abs(np.cross(self.point2 - self.point1, self.point1 - x_eval)) / norm(self.point2 - self.point1)


if __name__ == '__main__':
    point1 = np.array([[1], [0]])
    point2 = np.array([[0], [1]])
    point3 = np.array([[0], [0]])
    l = Line(point1, point2)
    print('distance is {}'.format(l.get_shortest_distance(point3)))