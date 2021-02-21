import numpy as np
import random
from utilities import calculate_optimal_threshold, calculate_VI_fraction, calculate_F1

"""
Implementation based on:
https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
written by Iran Macedo
"""

class Particle():
    def __init__(self):
        self.position = np.random.uniform(-2, 2, 6)
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.random.uniform(0, 0, 6)

    def move(self):
        self.position = self.position + self.velocity

class Space():

    def __init__(self, n_particles, X_train, Y_train):
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = []
        self.X_train = X_train
        self.Y_train = Y_train

    def fitness(self, particle):
        w_up = particle.position[:3]
        w_down = particle.position[3:]
        VI = calculate_VI_fraction(X=self.X_train, w_up=w_up, w_down=w_down)
        F1_score = calculate_F1(VI=VI, threshold=0.2, ground_truth=self.Y_train)
        return 1 - F1_score

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            if(self.gbest_value > fitness_cadidate):
                self.gbest_value = fitness_cadidate
                self.gbest_position = particle.position
                """
                print("---------------------------------------")
                print(f"Best position:{self.gbest_position}\n")
                print(f"Max F1_score:{1-self.gbest_value}\n")
                print("---------------------------------------")
                """

    def move_particles(self):
        W = 0.9
        c1 = 2
        c2 = 2
        for particle in self.particles:
            new_velocity = (W * particle.velocity) + (c1 * random.random()) * (particle.pbest_position - particle.position) + \
                            (random.random() * c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()
