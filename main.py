import matplotlib.pyplot as plt
import numpy as np
from math import sin
from math import cos
from math import pi
from math import sqrt


# Determines if a given point is inside an area, thus collision
def does_collide(coordinates, point, mark_point=False):
    global piste
    #global vektorix
    #global vektoriy
    
    if len(point) != 3:
        print("Collision point is not 3-dimensional!")
        pass

    last_coord = None
    for coord in coordinates:
        if last_coord is None:
            pass
        else:
            vector_n_plus1 = coord - last_coord  # Vector from coord_n to coord_n+1
            #vektorix = vector_n_plus1[0]
            #vektoriy = vector_n_plus1[1]
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
        piste = np.array([point[0], point[1], point[2]])
    return True

# Find nearest side of an object
# Return coordinates that make up the side


# Multiplies two 3-dimensional vectors
def vector_mult(vector1, vector2):
    result = [0, 0, 0]

    result[0] = vector1[1] * vector2[2] - vector1[2] * vector2[1]
    result[1] = vector1[2] * vector2[0] - vector1[0] * vector2[2]
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
def multiple_collision(coordinates1, coordinates2, cm1, cm2):
    # Check if any dots of shape 1 are inside shape 2
    for coord in coordinates1:
        if does_collide(coordinates2, coord, mark_point=True):
            return coord, coordinates2, cm1, cm2  # Return collision point and collision receiver

    # Check if any dots of shape 2 are inside shape 1   
    for coord in coordinates2:
        if does_collide(coordinates1, coord, mark_point=True):
            return coord, coordinates1, cm1, cm2  # Return collision point and collision receiver

    # We can be sure that no collision happened
    return None, None, None, None


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

def find_collision_normal(point1, point2, collision_point, cm_causer):

    # Create the vector from point1 to point2
    point1_to_point2 = point2 - point1

    # Create exactly two normals. One of them is the "right" normal
    normal1 = np.array([point1_to_point2[1], -point1_to_point2[0], 0])
    normal2 = np.array([-point1_to_point2[1], point1_to_point2[0], 0])

    # Create a vector from collision point to causer's mass center
    collision_point_to_cm_causer = cm_causer - collision_point

    # Check which normal has positive dot product with the collision point
    dotproduct1 = (collision_point_to_cm_causer[0] * normal1[0] +
                   collision_point_to_cm_causer[1] * normal1[1] +
                   collision_point_to_cm_causer[2] * normal1[2])

    if dotproduct1 > 0:
        return normal2

    dotproduct2 = (collision_point_to_cm_causer[0] * normal2[0] +
                   collision_point_to_cm_causer[1] * normal2[1] +
                   collision_point_to_cm_causer[2] * normal2[2])

    if dotproduct2 > 0:
        return normal1


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
    
    has_collided = False

    cm1 = [1 / 3 * (coordinates1[0][0] + coordinates1[1][0] + coordinates1[2][0]),
           1 / 3 * (coordinates1[0][1] + coordinates1[1][1] + coordinates1[2][1]), 0]
    cm2 = [1 / 4 * (sum(coordinates2[:, 0]) - coordinates2[-1][0]),
           1 / 4 * (sum(coordinates2[:, 1]) - coordinates2[-1][1]), 0]

    # Only gravity for now
    # At this scale, affecting acceleration forces stay constant
    a = [0, -9.81]

    # Define starting speed as a 2D array. With x and y components
    v1 = [start_velocity1 * cos(velocity_angle_radians1),
          start_velocity1 * sin(velocity_angle_radians1), 0]
    v2 = [start_velocity2 * cos(velocity_angle_radians2),
          start_velocity2 * sin(velocity_angle_radians2), 0]
    
    time = 0

    # Run for five seconds
    while time < 1.5:
        # Now calculate the changes to velocity
        # Velocity for every coordinate of a moving object is the same
        v1 = [v1[0] + a[0] * dt, v1[1] + a[1] * dt, 0]
        v2 = [v2[0] + a[0] * dt, v2[1] + a[1] * dt, 0]

        # Now calculate the changes to displacement
        # Displacement change will affect all the coords equally
        coordinates1, cm1 = kinematic_displacement(coordinates1, v1, dt, cm1)
        coordinates2, cm2 = kinematic_displacement(coordinates2, v2, dt, cm2)

        # Rotate objects. Rotation is scaled with time.
        coordinates1 = rotate_around_z(coordinates1, cm1, angular_velocity1 * dt)
        coordinates2 = rotate_around_z(coordinates2, cm2, angular_velocity2 * dt)

        

        # Plot at specific time intervals. Round the value by 2 digits for comparison precision.
        if round(time % plot_interval, 2) == 0:
            plt.plot(coordinates1[:, 0], coordinates1[:, 1])
            plt.plot(coordinates2[:, 0], coordinates2[:, 1])
            
        collision_point, collision_receiver, cm_receiver, cm_causer = multiple_collision(coordinates1, coordinates2, cm1, cm2)

        # Collision detection for blocks. If collides breaks the loop
        if collision_point is not None and has_collided == False:
            has_collided = True
            side_coord1, side_coord2 = find_nearest_side_to_point(collision_receiver, collision_point, True)
            
            
            
            vektorix = side_coord1[0] - side_coord2[0]
            vektoriy = side_coord1[1] - side_coord2[1]
            
            tx = vektorix/sqrt(vektorix * vektorix + vektoriy * vektoriy)
            ty = vektoriy/sqrt(vektorix * vektorix + vektoriy * vektoriy)
            
            nx = ty
            ny = tx
            
            ny = -(ny)
            
            #n = [nx, ny, 0]
            
            n = find_collision_normal(side_coord1, side_coord2, collision_point, cm_causer)
            #print(n)
            
            
            rAP = collision_point - cm1
            rBP = collision_point - cm2
            
            angular_velocity1_vector = [0, 0, angular_velocity1]
            angular_velocity2_vector = [0, 0, angular_velocity2]
            
            #print(rAPxn)
            
            wxrAP = vector_mult(angular_velocity1_vector, rAP)
            wxrBP = vector_mult(angular_velocity2_vector, rBP)
            vAP = [v1[0] + wxrAP[0], v1[1] + wxrAP[1]]
            #print(vAP)
            vBP = [v2[0] + wxrBP[0], v2[1] + wxrBP[1]]
            #print(vBP)
            vAB = [vAP[0] - vBP[0], vAP[1] - vBP[1], 0]
            
            yläpuoli = vAB[0] * n[0] + vAB[1] * n[1]
            #if yläpuoli > 0:
               # nx = -(nx)
               # ny = -(ny)
                #n = [nx, ny, 0]
                #yläpuoli = vAB[0] * n[0] + vAB[1] * n[1]
                
            #print(yläpuoli)
            
            rAPxn = vector_mult(rAP, n)
            rBPxn = vector_mult(rBP, n)
            
            #absrAPxn = [abs(rAPxn[0]), abs(rAPxn[1]), abs(rAPxn[2])]
            #absrBPxn = [abs(rBPxn[0]), abs(rBPxn[1]), abs(rBPxn[2])]
            
            absrAPxn2 = [abs(rAPxn[0])**2, abs(rAPxn[1])**2, abs(rAPxn[2])**2]
            absrBPxn2 = [abs(rBPxn[0])**2, abs(rBPxn[1])**2, abs(rBPxn[2])**2]
            
            e = 0.5
            
            ma = 2
            mb = 1
            ja = 0.02
            jb = 0.02
            
            alapuoli = 1/ma + 1/mb + absrAPxn2[2]/ja + absrBPxn2[2]/jb
            #print(alapuoli)
            
            impulssi = -(1+e) * yläpuoli / alapuoli
            
            v1l = [v1[0] + (impulssi/ma) * n[0], v1[1] + (impulssi/ma) * n[1]]
            v2l = [v2[0] + (impulssi/mb) * n[0], v2[1] + (impulssi/mb) * n[1]]
            
            angular_velocity_end1 = angular_velocity1 + (impulssi / ja) * (absrAPxn2[2])
            angular_velocity_end2 = angular_velocity2 + (impulssi / jb) * (absrBPxn2[2])
            
            # Now calculate the changes to velocity
            # Velocity for every coordinate of a moving object is the same
            v1 = [v1[0] + v1[0] + a[0] * dt, v1l[1] + v1[1] + a[1] * dt, 0]
            v2 = [v2[0] + v1[0] + a[0] * dt, v2l[1] + v2[1] + a[1] * dt, 0]

            # Now calculate the changes to displacement
            # Displacement change will affect all the coords equally
            
            #coordinates1, cm1 = kinematic_displacement(coordinates1, v1l, dt, cm1)
            #coordinates2, cm2 = kinematic_displacement(coordinates2, v2l, dt, cm2)

            # Rotate objects. Rotation is scaled with time.
            #coordinates1 = rotate_around_z(coordinates1, cm1, angular_velocity_end1 * dt)
            #coordinates2 = rotate_around_z(coordinates2, cm2, angular_velocity_end2 * dt)

            #time += dt
            
            print(impulssi)
            print(angular_velocity_end1, angular_velocity_end2)
            print(v1l, v2l)
            print("Collision at time", time)
            
        
        
        
        time += dt

    plt.axis('scaled')
    plt.show()


# test_rotate()
# test_kinematic(30, pi / 4)
# test_collision()

test_multiple(start_velocity1=15, velocity_angle_radians1=pi / 4,
              start_velocity2=9.3, velocity_angle_radians2=2 * pi / 3,
              angular_velocity1=pi/3, angular_velocity2=1,
              dt=0.01,
              plot_interval=0.0001)
