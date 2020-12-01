import matplotlib.pyplot as plt


def vector_mult(x=[0] * 2, y=[0] * 2, z=[0] * 2):
    i = y[0] * z[1] - z[0] * y[1]
    j = z[0] * x[1] - x[0] * z[1]
    k = x[0] * y[1] - y[0] * x[1]

    print(i)
    print(j)
    print(k)


def make_xy_map(v: list):
    x = []
    y = []
    z = []

    # Construct lists of individual components
    for component in v:
        x.append(component[0])
        y.append(component[1])
        z.append(component[2])

    return x, y, z


# Define dots (or vectors)
v1 = (-0.1, 0.2, 0)
v2 = (0.71, -0.71, 0)

x, y, z = make_xy_map([v1, v2])

# Makes the same list as
# x = [-0.1, 0.71]
# y = [0.2, -0.71]
# y = [0, 0]

print(x, y, z)
