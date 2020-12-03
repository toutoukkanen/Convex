import matplotlib.pyplot as plt
import numpy as np
from math import sin
from math import cos
from math import pi


# "Moves" origo to mass_center and calculates there.
# Then moves origo back and returns new values
def rotate_around_z(coordinates, mass_center, angle_radians):

    # Calculate coords relative to mass center
    coordinates[:, 0] = coordinates[:, 0] - mass_center[0]
    coordinates[:, 1] = coordinates[:, 1] - mass_center[1]

    # Rotate relative coordinates around z-axis
    for coord in coordinates:
        x: float = coord[0]
        y: float = coord[1]

        coord[0] = coord[0] * cos(angle_radians) - coord[1] * sin(angle_radians)
        coord[1] = x * sin(angle_radians) + y * cos(angle_radians)

    # Transform coordinates back to origo-centric format
    coordinates[:, 0] = coordinates[:, 0] + mass_center[0]
    coordinates[:, 1] = coordinates[:, 1] + mass_center[1]

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


# TODO: Make to work
# Displace given object by affecting velocity
def kinematic_displacement(x: list, y: list, v: list, dt):
    x_new = []
    y_new = []

    # x_new = [x_old + v[0] for x_old in x]
    # y_new = [(y_old + (v[0] * dt)) for y_old in y]

    # return x_new, y_new


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

    cm = (0.5, 0.5)  # Mass center

    rotate_radians = pi / 4

    print("Coordinate lists:\n", coordinates)

    plt.plot(coordinates[:, 0], coordinates[:, 1])

    coordinates = rotate_around_z(coordinates, cm, rotate_radians)

    plt.plot(coordinates[:, 0], coordinates[:, 1])
    plt.show()


def test_kinematic(start_velocity, velocity_angle_radians):
    # Define dots (or vectors)
    # v1 = (-0.1, 0.2, 0)
    # v2 = (0.71, -0.71, 0)

    # A triangle
    coordinates = np.array([
        (1.5, 0, 0),
        (-1, 0.5, 0),
        (-1, -0.5, 0),
        (1.5, 0, 0)  # Last dot to complete the triangle
    ])

    x, y, z = make_xyz_map(coordinates)

    # Only gravity for now
    # At this scale, affecting acceleration forces stay constant
    ax = 0
    ay = -9.81

    # Define starting speed as a 2D array. With x and y components
    # v = [ [start_velocity * cos(velocity_angle_radians), start_velocity * sin(velocity_angle_radians)] ]

    # vx = [v * cos(velocity_angle_radians)]
    # vy = [v * sin(velocity_angle_radians)]

    dt = 0.1  # Time interval

    # Now calculate the changes to velocity
    # v.append([[v[-1][0] + dt], [y[-1][1] + ay * dt]])
    # v.append([1,1])
    # print(v)

    # Now calculate the changes to displacement
    # Displacement change will affect all the coords equally
    # kinematic_displacement(x, y, vx, vy, dt)


# test_kinematic(30, pi / 4)
test_rotate()

