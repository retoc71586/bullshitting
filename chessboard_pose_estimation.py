import numpy as np
import cv2
import glob
import os
import os


class Legend:
    def __init__(
        self,
        legend_offset=20,
        legend_font=cv2.FONT_HERSHEY_SIMPLEX,
        legend_scale=0.5,
        legend_thickness=1,
        x_color=(0, 0, 255),
        y_color=(0, 255, 0),
        z_color=(255, 0, 0),
    ):
        self.legend_offset = legend_offset
        self.legend_font = legend_font
        self.legend_scale = legend_scale
        self.legend_thickness = legend_thickness
        self.x_color = x_color
        self.y_color = y_color
        self.z_color = z_color

    def draw_legend(self, drew_frame):
        cv2.putText(
            drew_frame,
            "X-axis",
            (self.legend_offset, self.legend_offset),
            self.legend_font,
            self.legend_scale,
            self.x_color,
            self.legend_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            drew_frame,
            "Y-axis",
            (self.legend_offset, self.legend_offset * 2),
            self.legend_font,
            self.legend_scale,
            self.y_color,
            self.legend_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            drew_frame,
            "Z-axis",
            (self.legend_offset, self.legend_offset * 3),
            self.legend_font,
            self.legend_scale,
            self.z_color,
            self.legend_thickness,
            cv2.LINE_AA,
        )


def createBoard():
    """
    Creates a Charuco board
    """
    # create a charuco board
    squaresX = 5
    squaresY = 7
    squareLength = 0.04
    markerLength = 0.02
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY), squareLength, markerLength, dictionary
    )
    frame = board.generateImage(
        outSize=(600, 500), marginSize=10, borderBits=1)
    cv2.imwrite("calib_data/charuco_board.png", frame)

    return frame


def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    visualise = 1
    legend = Legend()
    square_side_len = 0.03

    # Read input

    current_file_path = os.path.abspath(__file__)
    video_path = os.path.join(os.path.dirname(current_file_path), "1_checkerd_calib_high_res.mp4")
    cam = cv2.VideoCapture(video_path)
    valid, frame = cam.read()
    assert valid, "Failed to read from camera"
    h, w = frame.shape[:2]
    # Copy the original image a bunch of time for visualization of different steps
    img = frame.copy()
    distorted_image = frame.copy()
    world_coord_img = frame.copy()

    # corners refinement termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern_size = (11, 8)  # interior number of corners (row, col)
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float64)
    # fancy way for creating an array of dim (row, col, 3) where row[i] = i%8 and col[i] = i//8
    objp[:, :2] = np.mgrid[0: pattern_size[1],
                           0: pattern_size[0]].T.reshape(-1, 2)
    # actual size of lab's calib chessboard, in meters
    objp[:, :2] *= square_side_len
    objp[:, 2] = 0.006
    objp = objp.astype(np.float32)

    print("objp: ", objp)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    # Calibrate camera
    # For the high res vid
    mtx = np.array(
        [
            [3075.880615234375, 0.0, 1942.84619140625],
            [0.0, 3075.880615234375, 1103.8968505859375],
            [0.0, 0.0, 1.0],
        ]
    )
    # For ros low res
    # mtx = np.array([512.6467895507812, 0.0, 323.8077087402344, 0.0, 512.6467895507812, 183.98281860351562, 0.0,0.0,1.0,])
    dist = np.array([12.029667854309082, -115.97200775146484, 0.002035579876974225, 0.002628991613164544,
                    373.9336242675781, 11.838055610656738, -114.76853942871094, 369.8811340332031, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    frame = dst[y: y + h, x: x + w]
    undistorted_image = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        frame, pattern_size, cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE, )
    # If found, add object points, image points (after refining them)
    assert ret == True, print("Failed finding board")
    # objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    print("corners2: ", corners2.shape)
    # Draw and display the corners
    cv2.line(distorted_image, tuple(corners2[0][0].astype(int)), tuple(
        corners2[10][0].astype(int)), (0, 0, 255), 2,)
    cv2.line(undistorted_image, tuple(corners2[0][0].astype(int)), tuple(
        corners2[10][0].astype(int)), (0, 0, 255), 2,)
    cv2.imwrite("calib_data/distorted.png", distorted_image)
    cv2.imwrite("calib_data/undistorted.png", undistorted_image)

    cv2.drawChessboardCorners(frame, pattern_size, corners2, ret)
    cv2.imwrite("calib_data/detected_corners.png", frame)

    # Find camera pose
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, newcameramtx, dist)
    if visualise:
        legend.draw_legend(frame)
        drew_frame = cv2.drawFrameAxes(
            frame, newcameramtx, dist, rvecs, tvecs, square_side_len)
        output_file_path = os.path.join(
            os.path.dirname(__file__), "calib_data", "estimated_pose.png"
        )
        assert cv2.imwrite(output_file_path, drew_frame)
        # cv2.imshow('frame', drew_frame)
        # cv2.waitKey(0)
        # print("Closed window")

    # Calculate mean reprojection error
    imgpoints2, _ = cv2.projectPoints(objp, rvecs, tvecs, newcameramtx, dist)
    error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print("total error: {} px, below .33px is acceptable.".format(
        error))

    cv2.drawChessboardCorners(world_coord_img, pattern_size, imgpoints2, ret)
    cv2.imshow("world_coord_img", world_coord_img)
    cv2.waitKey(0)

    # for i in range(imgpoints2.shape[0]):
    #     cv2.putText(world_coord_img, str(objp[i]*100), tuple([imgpoints2[i][0][0].astype(
    #         int), imgpoints2[i][0][1].astype(int)]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,)
    # cv2.imshow("world_coord_img", world_coord_img)
    # cv2.waitKey(0)