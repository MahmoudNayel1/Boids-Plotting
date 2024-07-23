import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Boid:
    def __init__(self, x, y, vx, vy):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)


# Making boids move cohesively with each other by assigning cohesion force
def cohesion(boid, other_boids):
    if not other_boids:
        return np.zeros(2)

    center_of_mass = np.mean([other.position for other in other_boids], axis=0)
    distance = np.linalg.norm(center_of_mass - boid.position)
    if distance == 0:
        return np.zeros(2)
    cohesion_force = (center_of_mass - boid.position) / distance
    return cohesion_force


# Making boids separate from each other by assigning separation force 
def separation(boid, other_boids, min_distance=5.0):
    separation_force = np.zeros(2)
    for other in other_boids:
        distance = np.linalg.norm(boid.position - other.position)
        if distance < min_distance and distance != 0:
            separation_force += (boid.position - other.position) / distance
    return separation_force

# Making boids moving in agroup with nearly same velocity 
def alignment(boid, other_boids):
    if not other_boids:
        return np.zeros(2)

    average_velocity = np.mean([other.velocity for other in other_boids], axis=0)
    distance = np.linalg.norm(average_velocity - boid.velocity)
    if distance == 0:
        return np.zeros(2)
    alignment_force = (average_velocity - boid.velocity) / distance
    return alignment_force


def decision_making(boid, other_boids):
    # Implement decision-making using cohesion, separation, and alignment
    cohesion_force = cohesion(boid, other_boids)
    separation_force = separation(boid, other_boids)
    alignment_force = alignment(boid, other_boids)

    # Update boid's velocity based on the calculated forces
    boid.velocity += cohesion_force + separation_force + alignment_force

    # Update boid's position based on velocity
    boid.position += boid.velocity

    # Boundary check to keep boids within the frame
    boid.position = np.clip(boid.position, 0, 100)


# initializing boids with random position and velocities 
def initialize_boids(num_boids):
    boids = []
    for _ in range(num_boids):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
        boids.append(Boid(x, y, vx, vy))
    return boids

# saving the data of boids in CSV file 
def save_to_csv(boids, filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for boid in boids:
            writer.writerow([boid.position[0], boid.position[1], boid.velocity[0], boid.velocity[1]])


# 
def update_boids(boids):
    for boid in boids:
        # Get neighbors within a certain range 
        neighbors = [other for other in boids if np.linalg.norm(boid.position - other.position) < 30.0]

        # Calculate forces for cohesion, separation, and alignment
        cohesion_force = cohesion(boid, neighbors)
        separation_force = separation(boid, neighbors)
        alignment_force = alignment(boid, neighbors)

        # Update boid's velocity based on the calculated forces
        boid.velocity += cohesion_force + separation_force + alignment_force

        # Update boid's position based on velocity
        boid.position += boid.velocity

        # Boundary check to keep boids within the frame
        boid.position = np.clip(boid.position, 0, 100)


# For visualization
def animate(step, boids, sc):
    update_boids(boids)
    x = [boid.position[0] for boid in boids]
    y = [boid.position[1] for boid in boids]
    sc.set_offsets(np.c_[x, y])
    return sc,


# For running the whole simulation
def run_simulation(num_boids, steps, filename):
    boids = initialize_boids(num_boids)

    # Create a scatter plot for boid positions
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], marker='o')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    anim = FuncAnimation(fig, animate, fargs=(boids, sc), frames=steps, interval=50, blit=True)

    plt.show()

    # Save positions and velocities to CSV at the end of the simulation
    save_to_csv(boids, filename)

# Run simulation with 100 boids for 200 steps
run_simulation(100, 200, 'simulation_100_boids.csv')
run_simulation(10, 200, 'simulation_10_boids.csv')