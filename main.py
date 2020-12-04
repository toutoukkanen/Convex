import matplotlib.pyplot as plt
import numpy as np
from math import sin
from math import cos
from math import pi


# "Moves" origo to mass_center and calculates there.
# Then moves origo back and returns new values
def rotate_around_z(coordinates, cm, angle_radians):

    # Calculate coords relative to mass center
    coordinates[:, 0] = coordinates[:, 0] - cm[0]
    coordinates[:, 1] = coordinates[:, 1] - cm[1]

    # Rotate relative coordinates around z-axis
    for coord in coordinates:
        # Save old x because we want to access it when changing y
        x: float = coord[0]

        coord[0] = coord[0] * cos(angle_radians) - coord[1] * sin(angle_radians)
        coord[1] = x * sin(angle_radians) + coord[1] * cos(angle_radians)

    # Transform coordinates back to origo-centric format
    coordinates[:, 0] = coordinates[:, 0] + cm[0]
    coordinates[:, 1] = coordinates[:, 1] + cm[1]

    return coordinates


# Currently unused
def vector_mult(x=[0] * 2, y=[0] * 2, z=[0] * 2):  # 0 as default argument so no out of bounds
    # Example of vector multiplication
    # i, j, k = vector_mult(x, y, z)
    # print("v1 x v2 =", vector_mult(x,y,z))

    i = y[0] * z[1] - z[0] * y[1]
    j = z[0] * x[1] - x[0] * z[1]
    k = x[0] * y[1] - y[0] * x[1]

    return i, j, k


# Given a list of coordinates, return them in component lists of x,y,z
def make_xyz_map(coordinates: list):
    x = []
    y = []
    z = []

    # Construct lists of individual components
    for coord_tuple in coordinates:
        x.append(coord_tuple[0])
        y.append(coord_tuple[1])
        z.append(coord_tuple[2])

    return x, y, z


# Displace given object by affecting velocity
def kinematic_displacement(coordinates, v: list, dt, cm):

    # First update the new position of the mass center
    # using the first defined coordinate
    cm[0] = coordinates[0][0] + v[-1][0] * dt
    cm[1] = coordinates[0][1] + v[-1][1] * dt

    # Now update rest of the coordinates
    for coord in coordinates:
        coord[0] = coord[0] + v[-1][0] * dt
        coord[1] = coord[1] + v[-1][1] * dt

    # Force the function to give the updated center of the mass back along with coords
    return coordinates, cm


def test_rotate():
    # A triangle
    # coordinates = np.array([
    #     (1.5, 0, 0),
    #     (-1, 0.5, 0),
    #     (-1, -0.5, 0),
    #     (1.5, 0, 0)  # Last dot to complete the triangle
    # ])

    # A cube
    coordinates = np.array([
        (0., 0., 0.),
        (1., 0., 0.),
        (1., 1., 0.),
        (0., 1., 0.),
        (0., 0., 0.)
    ])

    # coordinates = np.array([
    #     (1.5, 0, 0),
    #     (-1, 0.5, 0),
    #     (-1, -0.5, 0),
    #     (1.5, 0, 0)  # Last dot to complete the triangle
    # ])

    cm_initial = (0.5, 0.5)  # Mass center of the cube when the bottom left dot is in origo

    rotate_radians = pi / 4

    print("Coordinate lists:\n", coordinates)

    plt.plot(coordinates[:, 0], coordinates[:, 1])

    coordinates = rotate_around_z(coordinates, cm_initial, rotate_radians)

    plt.plot(coordinates[:, 0], coordinates[:, 1])
    plt.show()


def test_kinematic(start_velocity, velocity_angle_radians):

    # A triangle
    coordinates = np.array([
        (1.5, 0, 0),
        (-1, 0.5, 0),
        (-1, -0.5, 0),
        (1.5, 0, 0)  # Last dot to complete the triangle
    ])

    # A cube
    # coordinates = np.array([
    #     (0., 0., 0.),
    #     (1., 0., 0.),
    #     (1., 1., 0.),
    #     (0., 1., 0.),
    #     (0., 0., 0.)
    # ])

    time = 0  # Passed time
    dt = 0.1  # Time interval
    cm = [0, 0]
    # Calculate the center of the mass
    cm[0] = 1 / 3 * coordinates[:, 0]
    cm[1] = 1 / 3 * coordinates[:, 1]

    # Only gravity for now
    # At this scale, affecting acceleration forces stay constant
    a = (0, -9.81)

    # Define starting speed as a 2D array. With x and y components
    v = []
    v.append([start_velocity * cos(velocity_angle_radians), start_velocity * sin(velocity_angle_radians)])

    while time < 4:
        # Now calculate the changes to velocity
        # Velocity for every coordinate of a moving object is the same
        v.append([v[-1][0] + dt, v[-1][1] + a[-1] * dt])

        # Now calculate the changes to displacement
        # Displacement change will affect all the coords equally
        coordinates, cm = kinematic_displacement(coordinates, v, dt, cm)
        coordinates = rotate_around_z(coordinates, cm, 0.1)

        plt.plot(coordinates[:, 0], coordinates[:, 1])

        time += dt

    plt.axis('scaled')
    plt.show()


# test_rotate()
test_kinematic(30, pi / 4)
