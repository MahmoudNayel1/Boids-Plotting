import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Boid:
    def __init__(self, x, y, vx, vy):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)


def cohesion(boid, other_boids):
    if not other_boids:
        return np.zeros(2)

    center_of_mass = np.mean([other.position for other in other_boids], axis=0)
    distance = np.linalg.norm(center_of_mass - boid.position)
    if distance == 0:
        return np.zeros(2)
    cohesion_force = (center_of_mass - boid.position) / distance
    return cohesion_force


def separation(boid, other_boids, min_distance=5.0):
    separation_force = np.zeros(2)
    for other in other_boids:
        distance = np.linalg.norm(boid.position - other.position)
        if distance < min_distance and distance != 0:
            separation_force += (boid.position - other.position) / distance
    return separation_force


def alignment(boid, other_boids):
    if not other_boids:
        return np.zeros(2)

    average_velocity = np.mean(
        [other.velocity for other in other_boids], axis=0)
    distance = np.linalg.norm(average_velocity - boid.velocity)
    if distance == 0:
        return np.zeros(2)
    alignment_force = (average_velocity - boid.velocity) / distance
    return alignment_force


def obstacle_avoidance(boid, obstacle_position, avoidance_distance=10.0):
    # Calculate the force to avoid the obstacle
    to_obstacle = obstacle_position - boid.position
    distance = np.linalg.norm(to_obstacle)

    if distance < avoidance_distance and distance != 0:
        avoidance_force = -to_obstacle / distance  # Move away from the obstacle
    else:
        avoidance_force = np.zeros(2)

    return avoidance_force


def decision_making(boid, other_boids, behaviors, obstacle_position):
    # Implement decision-making using selected behaviors
    cohesion_force = cohesion(boid, other_boids)
    separation_force = separation(boid, other_boids)
    alignment_force = alignment(boid, other_boids)
    obstacle_avoidance_force = obstacle_avoidance(boid, obstacle_position)

    # Adjust the weights of forces based on provided behaviors
    weights = {'cohesion': 2.0, 'separation': 1.0,
               'alignment': 2.0, 'obstacle_avoidance': 1.0}

    total_force = (
        weights['cohesion'] * cohesion_force +
        weights['separation'] * separation_force +
        weights['alignment'] * alignment_force +
        weights['obstacle_avoidance'] * obstacle_avoidance_force
    )

    # Update boid's velocity based on the calculated forces
    boid.velocity += total_force

    # Update boid's position based on velocity
    boid.position += boid.velocity

    # Boundary check to keep boids within the frame
    boid.position = np.clip(boid.position, 0, 100)


def update_boids(boids, obstacle_position):
    for boid in boids:
        # Get neighbors within a certain range
        neighbors = [other for other in boids if np.linalg.norm(
            boid.position - other.position) < 30.0]

        # Implement decision-making using cohesion, separation, alignment, and obstacle avoidance
        decision_making(boid, neighbors, [
                        'cohesion', 'separation', 'alignment', 'obstacle_avoidance'], obstacle_position)


def initialize_boids(num_boids):
    boids = []
    for _ in range(num_boids):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
        boids.append(Boid(x, y, vx, vy))
    return boids


def animate(step, boids, obstacle_position, sc, obstacle_sc):
    update_boids(boids, obstacle_position)
    x = [boid.position[0] for boid in boids]
    y = [boid.position[1] for boid in boids]
    sc.set_offsets(np.c_[x, y])
    obstacle_sc.set_offsets(obstacle_position)
    return sc, obstacle_sc


def save_to_csv(boids, filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for boid in boids:
            writer.writerow([boid.position[0], boid.position[1],
                            boid.velocity[0], boid.velocity[1]])


def run_simulation(num_boids, steps, filename):
    boids = initialize_boids(num_boids)
    obstacle_position = np.array([70, 70], dtype=float)

    # Create a scatter plot for boid positions and obstacle
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], marker='o', c='blue')
    obstacle_sc = ax.scatter(
        obstacle_position[0], obstacle_position[1], marker='o', c='red')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    anim = FuncAnimation(fig, animate, fargs=(
        boids, obstacle_position, sc, obstacle_sc), frames=steps, interval=50, blit=True)

    plt.show()

    # Save positions and velocities to CSV at the end of the simulation
    save_to_csv(boids, filename)


# Run simulation with 100 boids for 200 steps
run_simulation(100, 200, 'simulation_Explored_Model_100_boids.csv')
