import matplotlib.pyplot as plt
from math import sin
from math import cos
from math import pi


# "Moves" origo to mass_center and calculates there.
# Then moves origo back and returns new values
def rotate_around_z(x: list, y: list, mass_center, angle_radians):
    x_new = []
    y_new = []

    # Calculate coords relative to mass center
    x_relative = [x_origo - mass_center[0] for x_origo in x]
    y_relative = [y_origo - mass_center[1] for y_origo in y]

    # Now rotate around z-axis
    for valueX, valueY in zip(x_relative, y_relative):
        x_new.append(valueX * cos(angle_radians) - valueY * sin(angle_radians))
        y_new.append(valueX * sin(angle_radians) + valueY * cos(angle_radians))

    # Transform back origo-centric format
    x_new = [x_relative + mass_center[0] for x_relative in x_new]
    y_new = [y_relative + mass_center[1] for y_relative in y_new]

    # print("Origo", x)
    # print("Relative", x_relative)
    # print("New", x_new)

    return x_new, y_new


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


def main():
    # Define dots (or vectors)
    # v1 = (-0.1, 0.2, 0)
    # v2 = (0.71, -0.71, 0)

    # A triangle
    # coordinates = [
    #     (1.5, 0, 0),
    #     (-1, 0.5, 0),
    #     (-1, -0.5, 0),
    #     (1.5, 0, 0)  # Last dot to complete the triangle
    # ]

    # A cube
    coordinates = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 0)
    ]

    cm = (0.5, 0.5)  # Mass center

    rotate_radians = pi / 4

    x, y, z = make_xyz_map(coordinates)

    print("Coordinate lists:", x, y, z)

    plt.plot(x, y)

    x, y = rotate_around_z(x, y, cm, rotate_radians)

    plt.plot(x, y)
    plt.show()


main()



