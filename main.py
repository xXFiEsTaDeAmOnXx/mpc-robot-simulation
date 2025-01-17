import random
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from scipy.optimize import minimize


# Class representing a rectangular obstacle in the environment
class Obstacle:
    def __init__(self, x, y, width, height):
        """
        Initialize the obstacle with position and dimensions.
        :param x: X-coordinate of the bottom-left corner of the obstacle
        :param y: Y-coordinate of the bottom-left corner of the obstacle
        :param width: Width of the obstacle
        :param height: Height of the obstacle
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def distance_cost(self, robot_x, robot_y):
        """
        Compute the cost based on the robot's proximity to the obstacle.
        The cost increases as the robot gets closer.
        :param robot_x: Robot's x-coordinate
        :param robot_y: Robot's y-coordinate
        :return: Exponential cost based on the inverse of the distance
        """
        closest_x = max(self.x, min(robot_x, self.x + self.width))
        closest_y = max(self.y, min(robot_y, self.y + self.height))
        distance = np.sqrt((robot_x - closest_x) ** 2 + (robot_y - closest_y) ** 2)

        return np.exp(1 / distance)

    def draw(self, ax):
        """
        Draw the obstacle on the given matplotlib axis.
        A red rectangle represents the obstacle, and a yellow area indicates a safety margin.
        :param ax: Matplotlib axis to draw on
        """
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


# Class representing a robot with simple kinematics
class Robot:
    def __init__(self, x, y, theta, body_radius=0.5):
        """
        Initialize the robot's state.
        :param x: Initial x-coordinate
        :param y: Initial y-coordinate
        :param theta: Initial orientation in radians
        :param body_radius: Radius of the robot's circular body
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.body_radius = body_radius
        self.size = body_radius * 2
        self.trajectory = [(x, y)]

    def update_state(self, v, omega, dt):
        """
        Update the robot's position and orientation based on velocity and angular velocity.
        :param v: Linear velocity
        :param omega: Angular velocity
        :param dt: Time step
        """
        dx = v * np.cos(self.theta) * dt
        dy = v * np.sin(self.theta) * dt
        dtheta = omega * dt

        self.x += dx
        self.y += dy
        self.theta += dtheta
        self.trajectory.append((self.x, self.y))

    def draw(self, ax):
        """
        Draw the robot's body and orientation on the given matplotlib axis.
        :param ax: Matplotlib axis to draw on
        """
        corners = np.array(
            [
                [-self.size / 2, -self.size / 2],
                [self.size / 2, -self.size / 2],
                [self.size / 2, self.size / 2],
                [-self.size / 2, self.size / 2],
                [-self.size / 2, -self.size / 2],
            ]
        )

        # Apply rotation matrix to transform the robot's body based on its orientation
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


# Class implementing Model Predictive Control (MPC) for trajectory planning
class MPCController:
    def __init__(self, dt=0.1, prediction_horizon=10):
        """
        Initialize the MPC controller with parameters.
        :param dt: Time step for the controller
        :param prediction_horizon: Number of future steps to predict
        """
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.Q = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])  # State cost matrix
        self.R = np.eye(2) * 0.1  # Control input cost matrix

    def compute_control(self, robot, waypoints, obstacles):
        """
        Compute the optimal control inputs using MPC.
        :param robot: The robot object
        :param waypoints: List of waypoints for the robot to follow
        :param obstacles: List of obstacles to avoid
        :return: Optimal linear velocity and angular velocity
        """
        A = np.eye(3)

        def objective(u):
            """
            Objective function to minimize.
            Includes terms for state error, control effort, and obstacle avoidance.
            :param u: Control input vector
            :return: Cost
            """
            cost = 0.0
            x_next = np.array([robot.x, robot.y, robot.theta])

            for i in range(self.prediction_horizon):
                v_i = u[i]
                omega_i = u[i + self.prediction_horizon]
                u_curr = np.array([v_i, omega_i])

                # Update state using linearized system dynamics
                B = np.array(
                    [
                        [np.cos(x_next[2]) * self.dt, 0],
                        [np.sin(x_next[2]) * self.dt, 0],
                        [0, self.dt],
                    ]
                )
                x_next = A @ x_next + B @ u_curr

                # Add obstacle avoidance cost
                for obstacle in obstacles:
                    distance_cost = obstacle.distance_cost(x_next[0], x_next[1])
                    cost += distance_cost

                # Add waypoint tracking and control effort costs
                target = np.array(waypoints[0])
                error = x_next - target
                cost += error.T @ self.Q @ error
                cost += u_curr.T @ self.R @ u_curr

            return cost

        # Initialize control inputs with zeros
        u0 = np.zeros(2 * self.prediction_horizon)
        bounds = [(-5, 5)] * (2 * self.prediction_horizon)  # Velocity bounds

        # Solve the optimization problem
        result = minimize(objective, u0, method="SLSQP", bounds=bounds)
        return result.x[0], result.x[self.prediction_horizon]


class RobotSimulation:
    def __init__(self, robot, waypoints, obstacles):
        # Initialize the simulation with the robot, waypoints, and obstacles
        self.robot = robot
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.window = None  # Tkinter window for visualization
        self.is_finished = False  # Tracks if the simulation is complete
        self.controller = MPCController()  # Initialize the MPC controller

        # Setup Matplotlib figure and axes
        self.fig, self.ax = plt.subplots()
        self.setup_plot()

    def setup_plot(self):
        # Configure the plot with axis settings and gridlines
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.invert_yaxis()  # Invert y-axis for a natural top-down view
        self.ax.set_xticks(np.arange(0, 21, 1))
        self.ax.set_yticks(np.arange(0, 21, 1))
        self.ax.grid(True)
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)

        # Plot the robot's trajectory and waypoints
        points = [(self.robot.x, self.robot.y)] + [
            (wx, wy) for wx, wy, _ in self.waypoints
        ]
        for i in range(len(points) - 1):
            self.ax.plot(
                [points[i][0], points[i + 1][0]],
                [points[i][1], points[i + 1][1]],
                "k--",  # Dashed line for trajectory
            )

    def update(self, frame):
        # Update the simulation for the current frame
        if not self.waypoints:
            return  # Do nothing if all waypoints are achieved

        # Compute control inputs using MPC
        v, omega = self.controller.compute_control(
            self.robot, self.waypoints, self.obstacles
        )

        # Update the robot's state with the computed inputs
        self.robot.update_state(v, omega, self.controller.dt)

        # Check if the robot is close enough to the current waypoint
        if (
            np.sqrt(
                (self.waypoints[0][0] - self.robot.x) ** 2
                + (self.waypoints[0][1] - self.robot.y) ** 2
            )
            < 0.4
            and np.abs(self.robot.theta - self.waypoints[0][2]) < 0.2
        ):
            self.waypoints.pop(0)  # Remove the reached waypoint

            if len(self.waypoints) == 0:
                self.check_finish()  # Check if the simulation is complete

        # Clear and redraw the plot
        self.ax.clear()
        self.setup_plot()

        # Draw the robot and obstacles
        self.robot.draw(self.ax)
        for obstacle in self.obstacles:
            obstacle.draw(self.ax)

        # Draw waypoints and their orientation as arrows
        for wx, wy, wpsi in self.waypoints:
            self.ax.plot(wx, wy, "go")  # Green circle for waypoint
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
        # Mark the simulation as finished and display a message
        if not self.waypoints and not self.is_finished:
            self.is_finished = True
            print("We achieved all waypoints! Finishing simulation...")
            self.window.after(1000, self.close_simulation)  # Delay closure

    def close_simulation(self):
        # Close the Tkinter window and the Matplotlib plot
        self.window.quit()
        self.window.destroy()
        plt.close("all")

    def run(self, total_time=20, dt=0.1):
        # Run the simulation with specified total time and time step
        times = np.arange(0, total_time, dt)
        self.anim = FuncAnimation(
            self.fig, self.update, frames=times, repeat=False, blit=False
        )

        # Create and configure the Tkinter window
        self.window = tk.Tk()
        self.window.title("Trajectory Simulation with Obstacles")

        # Embed Matplotlib figure into Tkinter using FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Set the window size and position
        self.window.geometry("800x600+100+100")
        self.window.mainloop()

    def save_gif(self, filename="robot_simulation.gif", total_time=20, dt=0.1):
        # Save the simulation as a GIF file
        times = np.arange(0, total_time, dt)
        anim = FuncAnimation(
            self.fig, self.update, frames=times, repeat=False, blit=False
        )
        anim.save(filename, writer="imagemagick", fps=30)


def main():
    # Main function to initialize and run the simulation
    initial_x = 1.0  # Initial x-coordinate of the robot
    initial_y = 1.0  # Initial y-coordinate of the robot
    initial_orientation = np.radians(45)  # Initial orientation in radians

    robot = Robot(initial_x, initial_y, initial_orientation)

    # Generate random waypoints for the robot to follow
    waypoints = [
        (
            random.randint(2, 16),
            random.randint(2, 16),
            np.deg2rad(random.randint(0, 360)),
        )
        for _ in range(5)
    ]

    # Generate random obstacles in the environment
    obstacles = [
        Obstacle(
            random.randint(2, 16),
            random.randint(2, 16),
            random.randint(1, 5),
            random.randint(1, 5),
        )
        for _ in range(2)
    ]

    # Create and run the robot simulation
    simulation = RobotSimulation(robot, waypoints, obstacles)
    simulation.run()


if __name__ == "__main__":
    main()
