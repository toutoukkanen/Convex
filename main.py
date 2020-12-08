import matplotlib.pyplot as plt
import numpy as np
from math import sin
from math import cos
from math import pi
from math import sqrt


# Determines if a given point is inside an area, thus collision
def does_collide(coordinates, point, mark_point=False):
    if len(point) != 3:
        print("Collision point is not 3-dimensional!")
        pass

    last_coord = None

    for coord in coordinates:
        if last_coord is None:
            last_coord = coord
        else:
            vector_n_plus1 = coord - last_coord  # Vector from coord_n to coord_n+1
            vector_np = point - last_coord  # Vector from coord_n to point
            result = vector_mult(vector_n_plus1, vector_np)

            # Only if z-component is negative
            if result[2] < 0:
                return False

        last_coord = coord

    # At this point all z-components of multiplications are zero or positive
    # Thus we can be sure that collision happened
    if mark_point:
        plt.plot(point[0], point[1], "ro")
    return True


# Multiplies two 3-dimensional vectors
def vector_mult(vector1, vector2):
    result = [0, 0, 0]

    result[0] = vector1[1] * vector2[2] - vector1[2] * vector2[1]
    result[1] = -(vector1[0] * vector2[2] - vector1[2] * vector2[0])
    result[2] = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    return result


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

        # Rotation matrix
        coord[0] = coord[0] * cos(angle_radians) - coord[1] * sin(angle_radians)
        coord[1] = x * sin(angle_radians) + coord[1] * cos(angle_radians)

    # Transform coordinates back to origo-centric format
    coordinates[:, 0] = coordinates[:, 0] + cm[0]
    coordinates[:, 1] = coordinates[:, 1] + cm[1]

    return coordinates


# Given a list of coordinates, return them in component lists of x,y,z
# Unused, but might be useful in some circumstances
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
    cm[0] = cm[0] + v[0] * dt
    cm[1] = cm[1] + v[1] * dt

    # Now update rest of the coordinates
    for coord in coordinates:
        coord[0] = coord[0] + v[0] * dt
        coord[1] = coord[1] + v[1] * dt

    # Force the function to give the updated center of the mass back along with coords
    return coordinates, cm


# Checks if 2 object collide
# Return coordinate of collision and collision receiver
def multiple_collision(coordinates1, coordinates2):
    # Check if any dots of shape 1 are inside shape 2
    for coord in coordinates1:
        if does_collide(coordinates2, coord, mark_point=True):
            return coord, coordinates2  # Return collision point and collision receiver

    # Check if any dots of shape 2 are inside shape 1   
    for coord in coordinates2:
        if does_collide(coordinates1, coord, mark_point=True):
            return coord, coordinates1  # Return collision point and collision receiver

    # We can be sure that no collision happened
    return None, None


# Find nearest side of an object
# Return coordinates that make up the side
def find_nearest_side_to_point(coordinates, point, mark_points=False):

    lowest_distance = 0
    coord1 = [0, 0]
    coord2 = [0, 0]

    last_coord = None

    for coord in coordinates:
        if last_coord is None:
            pass
        else:
            numerator = abs((coord[0]-last_coord[0]) * (point[1]-last_coord[1]) -
                            (point[0]-last_coord[0]) * (coord[1]-last_coord[1]))
            denominator = sqrt((coord[0] - last_coord[0])**2 + (coord[1] - last_coord[1])**2)

            distance = numerator/denominator

            if lowest_distance == 0:  # Make sure that last_coord is valid
                lowest_distance = distance

            if distance < lowest_distance:
                lowest_distance = distance
                coord1 = coord
                coord2 = last_coord

        last_coord = coord

    if mark_points:
        plt.plot(coord1[0], coord1[1], "ro")
        plt.plot(coord2[0], coord2[1], "ro")

    return coord1, coord2


def calculate_collision_effects(coordinates1, coordinates2, cm1, cm2, v1, v2, av1, av2):
    # Calculate impulse of collision

    pass


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

    cm_initial = (0.5, 0.5)  # Mass center of the cube when the bottom left dot is in origo

    rotate_radians = pi / 4

    print("Coordinate lists:\n", coordinates)

    plt.plot(coordinates[:, 0], coordinates[:, 1])

    coordinates = rotate_around_z(coordinates, cm_initial, rotate_radians)

    plt.plot(coordinates[:, 0], coordinates[:, 1])
    plt.show()


def test_collision():
    # A triangle
    coordinates = np.array([
        (1.5, 0, 0),
        (-1, 0.5, 0),
        (-1, -0.5, 0),
        (1.5, 0, 0)  # Last dot to complete the triangle
    ])

    dot = [1, 0, 0]

    plt.plot(coordinates[:, 0], coordinates[:, 1])
    plt.plot(dot[0], dot[1], "ro")
    plt.axis('scaled')
    plt.show()

    # dot1 = [0.2, 0.05, 0]
    # dot2 = [0, 0, 5]
    # x = [-0.1, 0.71]
    # y = [0.2, -0.71]
    # print(vector_mult(x, y, [0, 0]))

    # Check if dot is inside our object
    print(does_collide(coordinates, dot))


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
    dt = 0.01  # Time interval
    cm = [0, 0]

    # Calculate the center of the mass
    # This bit of code is specifically for triangles
    cm = [1 / 3 * (coordinates[0][0] + coordinates[1][0] + coordinates[2][0]),
          1 / 3 * (coordinates[0][1] + coordinates[1][1] + coordinates[2][1])]

    # Only gravity for now
    # At this scale, affecting acceleration forces stay constant
    a = [0, -9.81]

    # Define starting speed as a 2D array. With x and y components
    v = [start_velocity * cos(velocity_angle_radians), start_velocity * sin(velocity_angle_radians)]

    while time < 5:
        # Now calculate the changes to velocity
        # Velocity for every coordinate of a moving object is the same
        v = [v[0] + a[0] * dt, v[1] + a[1] * dt]

        # Now calculate the changes to displacement
        # Displacement change will affect all the coords equally
        coordinates, cm = kinematic_displacement(coordinates, v, dt, cm)

        coordinates = rotate_around_z(coordinates, cm, 0.1)

        plt.plot(coordinates[:, 0], coordinates[:, 1])

        time += dt

    plt.axis('scaled')
    plt.show()


def test_multiple(start_velocity1=0., velocity_angle_radians1=0.,
                  start_velocity2=0., velocity_angle_radians2=0.,
                  angular_velocity1=0., angular_velocity2=0.,
                  dt=1.,
                  plot_interval=0.1):
    # A triangle
    coordinates1 = np.array([
        (1.5, 0, 0),
        (-1, 0.5, 0),
        (-1, -0.5, 0),
        (1.5, 0, 0)  # Last dot to complete the triangle
    ])

    # A cube
    coordinates2 = np.array([
        (10., 0., 0.),
        (11., 0., 0.),
        (11., 1., 0.),
        (10., 1., 0.),
        (10., 0., 0.)
    ])

    # Calculate the center of the mass
    cm1 = [1 / 3 * (coordinates1[0][0] + coordinates1[1][0] + coordinates1[2][0]),
           1 / 3 * (coordinates1[0][1] + coordinates1[1][1] + coordinates1[2][1])]
    cm2 = [1 / 4 * (sum(coordinates2[:, 0]) - coordinates2[-1][0]),
           1 / 4 * (sum(coordinates2[:, 1]) - coordinates2[-1][1])]

    # Only gravity for now
    # At this scale, affecting acceleration forces stay constant
    a = [0, -9.81]

    # Define starting speed as an array with x and y components
    v1 = [start_velocity1 * cos(velocity_angle_radians1),
          start_velocity1 * sin(velocity_angle_radians1)]
    v2 = [start_velocity2 * cos(velocity_angle_radians2),
          start_velocity2 * sin(velocity_angle_radians2)]

    time = 0  # Passed time

    # Run for five seconds unless collision happened
    while time < 5:
        # Now calculate the changes to velocity
        # Velocity for every coordinate of a moving object is the same
        v1 = [v1[0] + a[0] * dt, v1[1] + a[1] * dt]
        v2 = [v2[0] + a[0] * dt, v2[1] + a[1] * dt]

        # Now calculate the changes to displacement
        # Displacement change will affect all the coords equally
        coordinates1, cm1 = kinematic_displacement(coordinates1, v1, dt, cm1)
        coordinates2, cm2 = kinematic_displacement(coordinates2, v2, dt, cm2)

        # Rotate objects. Rotation is scaled with time.
        coordinates1 = rotate_around_z(coordinates1, cm1, angular_velocity1 * dt)
        coordinates2 = rotate_around_z(coordinates2, cm2, angular_velocity2 * dt)

        time += dt

        # Plot at specific time intervals. Round the value by 2 digits for comparison precision.
        if round(time % plot_interval, 2) == 0:
            plt.plot(coordinates1[:, 0], coordinates1[:, 1])
            plt.plot(coordinates2[:, 0], coordinates2[:, 1])

        collision_point, collision_receiver = multiple_collision(coordinates1, coordinates2)

        # Detect collision for objects
        if collision_point is not None:
            print("Collision at time", time)

            side_coord1, side_coord2 = find_nearest_side_to_point(collision_receiver, collision_point)

            # Determine new speed and angular speed

            break

    plt.axis('scaled')
    plt.show()


# test_rotate()
# test_kinematic(30, pi / 4)
# test_collision()

test_multiple(start_velocity1=15, velocity_angle_radians1=pi / 4,
              start_velocity2=10, velocity_angle_radians2=2 * pi / 3,
              angular_velocity1=pi / 2, angular_velocity2=1,
              dt=0.001,
              plot_interval=0.01)
