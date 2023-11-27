import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


np.set_printoptions(suppress=True)


############################################################################################

T_robot_robot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Transforms from tag frame to robot
T_robot_april_tag = np.array(
    [
        [1.0, 0.0, 0.0, 0.699],
        [0.0, 1.0, 0.0, -0.012],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Transforms from tag frame to camera frame
T_camera_tag = np.array(
    [
        [-0.99784732, 0.00869479, 0.06500096, -0.88385989],
        [0.00953968, 0.99987386, 0.01269893, -0.37047954],
        [-0.06488235, 0.01329168, -0.9978044, 1.83498459],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000],
    ]
)

T_camera_tag_new = np.array(
    [
        [0.99818706, 0.02746303, -0.05355721, -0.55566303],
        [0.02651187, -0.99947933, -0.01839006, -0.34514479],
        [-0.05403437, 0.01693682, -0.99839543, 1.84810026],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000],
    ]
)

# T_camera_tag = np.array(
#     [
#         [0.99872828, -0.02787592, 0.0420089, 0.30136755],
#         [-0.02830279, -0.99955329, 0.00960103, 0.02657498],
#         [0.0417225, -0.01077779, -0.9990711, 0.9233411],
#         [0.00000000, 0.00000000, 0.00000000, 1.00000000],
#     ]
# )

T_robot_camera = np.matmul(T_robot_april_tag, np.linalg.inv(T_camera_tag))
T_camera_robot = np.linalg.inv(T_robot_camera)

############################################################################################


def plot_frame(ax, frame, label):
    # x
    ax.quiver(
        frame[0, 3],
        frame[1, 3],
        frame[2, 3],
        frame[0, 0],
        frame[1, 0],
        frame[2, 0],
        color="r",
        length=0.2,
        normalize=True,
        zorder=2,
    )
    # y
    ax.quiver(
        frame[0, 3],
        frame[1, 3],
        frame[2, 3],
        frame[0, 1],
        frame[1, 1],
        frame[2, 1],
        color="g",
        length=0.2,
        normalize=True,
        zorder=2,
    )
    # z
    ax.quiver(
        frame[0, 3],
        frame[1, 3],
        frame[2, 3],
        frame[0, 2],
        frame[1, 2],
        frame[2, 2],
        color="b",
        length=0.2,
        normalize=True,
        zorder=2,
    )

    ax.text(frame[0, 3], frame[1, 3], frame[2, 3], label, fontsize=12)


def plot_rectangle(ax):
    # Define the rectangle vertices
    # [0, -0.5, 0], [1, -0.5, 0], [1, 0.5, 0], [0, 0.5, 0]])
    # X = np.array([0, 1, 1, 0])-0.699
    # Y = np.array([-0.5, -0.5, 0.5, 0.5])
    # X, Y = np.meshgrid(X, Y)
    # Z = np.array([0, 0, 0, 0])
    # Z = np.meshgrid(Z)
    # R = np.sqrt(X**2 + Y**2)
    # Z = 0*R-0.1

    # # Plot the rectangle
    # ax.plot_surface(X, Y, Z, color='burlywood', alpha=1, zorder=1, shade=False)

    x = np.array([0, 1, 1, 0]) - 0.699
    y = np.array([-0.5, -0.5, 0.5, 0.5])
    z = np.array([0, 0, 0, 0]) - 0.1

    surfaces = []

    surfaces.append([list(zip(x, y, z))])

    for surface in surfaces:
        ax.add_collection3d(
            Poly3DCollection(surface, color="burlywood", alpha=1, zorder=1, shade=False)
        )


# Define frames (expressed all in tag frame)
camera = np.eye(4)  # Camera frame expressed in tag frame
tag = T_camera_tag  # Tag frame expressed in tag frame
tag_new = T_camera_tag_new
robot = T_camera_robot

tag_new = np.eye(4)
camera = np.linalg.inv(T_camera_tag_new)
tag = np.linalg.inv(T_camera_tag_new)@T_camera_tag

# Plot the frames
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

plot_frame(ax, camera, "camera")
plot_frame(ax, tag, "tag")
plot_frame(ax, tag_new, "tag_new")
# plot_frame(ax, robot, "robot")

ax.legend(labels=["x", "y", "z"], loc="upper right", bbox_to_anchor=(1, 1))

plt.show()
