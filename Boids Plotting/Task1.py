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