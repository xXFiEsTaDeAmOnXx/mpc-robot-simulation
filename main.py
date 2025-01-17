import random
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from scipy.optimize import minimize


class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def distance_cost(self, robot_x, robot_y):
        closest_x = max(self.x, min(robot_x, self.x + self.width))
        closest_y = max(self.y, min(robot_y, self.y + self.height))
        distance = np.sqrt((robot_x - closest_x) ** 2 + (robot_y - closest_y) ** 2)

        return np.exp(1 / distance)

    def draw(self, ax):
        rect = Rectangle(
            (self.x, self.y),
            self.width,
            self.height,
            facecolor="red",
            alpha=0.5,
        )
        ax.add_patch(rect)

        safety_rect = Rectangle(
            (self.x - 2, self.y - 2),
            self.width + 4,
            self.height + 4,
            facecolor="yellow",
            alpha=0.2,
            linestyle="--",
            edgecolor="orange",
        )
        ax.add_patch(safety_rect)


class Robot:
    def __init__(self, x, y, theta, body_radius=0.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.body_radius = body_radius
        self.size = body_radius * 2
        self.trajectory = [(x, y)]

    def update_state(self, v, omega, dt):
        dx = v * np.cos(self.theta) * dt
        dy = v * np.sin(self.theta) * dt
        dtheta = omega * dt

        self.x += dx
        self.y += dy
        self.theta += dtheta
        self.trajectory.append((self.x, self.y))

    def draw(self, ax):
        corners = np.array(
            [
                [-self.size / 2, -self.size / 2],
                [self.size / 2, -self.size / 2],
                [self.size / 2, self.size / 2],
                [-self.size / 2, self.size / 2],
                [-self.size / 2, -self.size / 2],
            ]
        )

        rotation_matrix = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )
        transformed_corners = np.dot(corners, rotation_matrix.T) + np.array(
            [self.x, self.y]
        )
        ax.plot(transformed_corners[:, 0], transformed_corners[:, 1], "r-")


class MPCController:
    def __init__(self, dt=0.1, prediction_horizon=20):
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.Q = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        self.R = np.eye(2) * 0.1

    def compute_control(self, robot, waypoints, obstacles):
        A = np.eye(3)

        def objective(u):
            cost = 0.0
            x_next = np.array([robot.x, robot.y, robot.theta])

            for i in range(self.prediction_horizon):
                v_i = u[i]
                omega_i = u[i + self.prediction_horizon]
                u_curr = np.array([v_i, omega_i])

                B = np.array(
                    [
                        [np.cos(x_next[2]) * self.dt, 0],
                        [np.sin(x_next[2]) * self.dt, 0],
                        [0, self.dt],
                    ]
                )
                x_next = A @ x_next + B @ u_curr

                for obstacle in obstacles:
                    distance_cost = obstacle.distance_cost(x_next[0], x_next[1])
                    cost += distance_cost

                target = np.array(waypoints[0])
                error = x_next - target
                cost += error.T @ self.Q @ error
                cost += u_curr.T @ self.R @ u_curr

            return cost

        u0 = np.zeros(2 * self.prediction_horizon)
        bounds = [(-5, 5)] * (2 * self.prediction_horizon)

        result = minimize(objective, u0, method="SLSQP", bounds=bounds)
        return result.x[0], result.x[self.prediction_horizon]


class RobotSimulation:
    def __init__(self, robot, waypoints, obstacles):
        self.robot = robot
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.window = None
        self.is_finished = False
        self.controller = MPCController()

        self.fig, self.ax = plt.subplots()
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(0, 21, 1))
        self.ax.set_yticks(np.arange(0, 21, 1))
        self.ax.grid(True)
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)

        points = [(self.robot.x, self.robot.y)] + [
            (wx, wy) for wx, wy, _ in self.waypoints
        ]
        for i in range(len(points) - 1):
            self.ax.plot(
                [points[i][0], points[i + 1][0]],
                [points[i][1], points[i + 1][1]],
                "k--",
            )

    def update(self, frame):
        if not self.waypoints:
            return

        v, omega = self.controller.compute_control(
            self.robot, self.waypoints, self.obstacles
        )

        self.robot.update_state(v, omega, self.controller.dt)

        if (
            np.sqrt(
                (self.waypoints[0][0] - self.robot.x) ** 2
                + (self.waypoints[0][1] - self.robot.y) ** 2
            )
            < 0.4
            and np.abs(self.robot.theta - self.waypoints[0][2]) < 0.2
        ):
            self.waypoints.pop(0)

            if len(self.waypoints) == 0:
                self.check_finish()

        self.ax.clear()
        self.setup_plot()

        self.robot.draw(self.ax)
        for obstacle in self.obstacles:
            obstacle.draw(self.ax)

        for wx, wy, wpsi in self.waypoints:
            self.ax.plot(wx, wy, "go")
            arrow_length = 1.0
            self.ax.arrow(
                wx,
                wy,
                arrow_length * np.cos(wpsi),
                arrow_length * np.sin(wpsi),
                color="g",
                head_width=0.2,
                head_length=0.3,
            )

    def check_finish(self):
        if not self.waypoints and not self.is_finished:
            self.is_finished = True
            print("We achieved all waypoints! Finishing simulation...")
            self.window.after(1000, self.close_simulation)

    def close_simulation(self):
        self.window.quit()
        self.window.destroy()
        plt.close("all")

    def run(self, total_time=20, dt=0.1):
        times = np.arange(0, total_time, dt)
        self.anim = FuncAnimation(
            self.fig, self.update, frames=times, repeat=False, blit=False
        )

        self.window = tk.Tk()
        self.window.title("Trajectory Simulation with Obstacles")

        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.window.geometry("800x600+100+100")
        self.window.mainloop()

    def save_gif(self, filename="robot_simulation.gif", total_time=20, dt=0.1):
        times = np.arange(0, total_time, dt)
        anim = FuncAnimation(
            self.fig, self.update, frames=times, repeat=False, blit=False
        )
        anim.save(filename, writer="imagemagick", fps=30)


def main():
    initial_x = 1.0
    initial_y = 1.0
    initial_orientation = np.radians(45)

    robot = Robot(initial_x, initial_y, initial_orientation)

    waypoints = [
        (
            random.randint(2, 16),
            random.randint(2, 16),
            np.deg2rad(random.randint(0, 360)),
        )
        for _ in range(5)
    ]

    obstacles = [
        Obstacle(
            random.randint(2, 16),
            random.randint(2, 16),
            random.randint(1, 5),
            random.randint(1, 5),
        )
        for _ in range(2)
    ]

    simulation = RobotSimulation(robot, waypoints, obstacles)
    simulation.save_gif()


if __name__ == "__main__":
    main()
