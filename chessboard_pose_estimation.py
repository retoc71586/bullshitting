import numpy as np
import cv2
import glob
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

class PoseEstimator():
    def estimatePose(visualise = False):
        '''@return rvecs, tvecs'''
        np.set_printoptions(suppress=True)
        legend = Legend()
        square_side_len = 0.03
        board_height = 0.006
        board_height = 0.0
        undistort = False # I don't know why but yelds worse inliners percentage if true

        # Read input

        current_file_path = os.path.abspath(__file__)
        video_path = os.path.join(os.path.dirname(current_file_path), "input/1_checkerd_calib_high_res.mp4")
        cam = cv2.VideoCapture(video_path)
        valid, frame = cam.read()
        assert valid, "Failed to read video stream from file"
        h, w = frame.shape[:2]
        # Copy the original image a bunch of time for visualization of different steps
        img = frame.copy()
        distorted_image = frame.copy()
        dimentions_check = frame.copy()
        corners_idx = frame.copy()
        # corners refinement termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        (rows, cols) = (8, 11)  # interior number of corners (row, col)
        objp = []
        for x in range(cols):
            for y in range(rows):
                objp.append([x*square_side_len, y*square_side_len, board_height])
        objp = np.array(objp, np.float32)

        # Calibrate camera
        # For the high res vid
        mtx = np.array([
                [3075.880615234375, 0.0, 1942.84619140625],
                [0.0, 3075.880615234375, 1103.8968505859375],
                [0.0, 0.0, 1.0],
            ])
        # For ros low res
        # mtx = np.array([512.6467895507812, 0.0, 323.8077087402344, 0.0, 512.6467895507812, 183.98281860351562, 0.0,0.0,1.0,])
        dist = np.array([12.029667854309082, -115.97200775146484, 0.002035579876974225, 0.002628991613164544,
                        373.9336242675781, 11.838055610656738, -114.76853942871094, 369.8811340332031, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if undistort:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h))
            # undistort
            dst = cv2.undistort(frame, mtx, dist)
            # crop the image
            x, y, w, h = roi
            frame = dst[y: y + h, x: x + w]
            undistorted_image = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray, (rows, cols), cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        assert ret == True, print("Failed finding board")
        # objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(rows*cols, 2)

        # Find camera pose
        ret, rvecs, tvecs, inliners = cv2.solvePnPRansac(objp, corners2, mtx, dist, confidence=0.999)
        assert ret == True, print("Failed solving PnP")
        print(f"Inliners percentage: {len(inliners)/len(corners2)*100:.3f} % of point correspondences")
        # Create a 4x4 transformation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvecs)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvecs.flatten()
        print("Transformation matrix of chessboard pose in camera frame: \n",    transformation_matrix,)
        if visualise:
            # Draw image distortion
            cv2.line(distorted_image, tuple(corners2[0].astype(int)), tuple(
                corners2[rows-1].astype(int)), (0, 0, 255), 2,)
            cv2.imwrite("calib_data/distorted.png", distorted_image)
            if undistort:
                cv2.line(undistorted_image, tuple(corners2[0].astype(int)), tuple(
                corners2[rows-1].astype(int)), (0, 0, 255), 2,)
                cv2.imwrite("calib_data/undistorted.png", undistorted_image)

            # Draw circles at each corner with their index
            for i, (x, y) in enumerate(corners2):
                cv2.circle(corners_idx, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw a green circle
                cv2.putText(corners_idx, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imwrite("calib_data/corners_idx.png", corners_idx)

            # Draw and display the corners            
            cv2.drawChessboardCorners(frame, (rows, cols), corners2, ret)
            cv2.imwrite("calib_data/detected_corners.png", frame)
            legend.draw_legend(frame)
            drew_frame = cv2.drawFrameAxes(
                frame, mtx, dist, rvecs, tvecs, square_side_len)
            output_file_path = os.path.join(
                os.path.dirname(__file__), "calib_data", "estimated_pose.png"
            )
            assert cv2.imwrite(output_file_path, drew_frame)

        # Calculate mean reprojection error
        imgpoints2, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
        imgpoints2 = imgpoints2.reshape(rows*cols, 2)
        error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        print("total error: {} px, below .33px is acceptable.".format(
            error))
        
        side = np.array([0., square_side_len, 0., 1.])
        center = np.array([0., 0., 0., 1.])
        print("center[:3]", center[:3])
        cam_side = np.matmul(transformation_matrix, side)
        cam_center = np.matmul(transformation_matrix, center)
        print("cam_side: ",cam_side)
        print("cam_center: ",cam_center)
        px_side, _ = cv2.projectPoints(cam_side[:3], rvecs, tvecs, mtx, dist)
        px_side = px_side.reshape(2)
        px_center, _ = cv2.projectPoints(center[:3], rvecs, tvecs, mtx, dist)
        px_center = px_center.reshape(2)

        manual_px_center, _ = cv2.projectPoints(cam_center[:3], np.zeros(3),  np.zeros(3), mtx, dist)
        manual_px_center = manual_px_center.reshape(2)
        # cv2.line(dimentions_check, tuple([0,0]), tuple(px_side[:2].astype(int)), (0, 0, 255), 2,)
        cv2.circle(dimentions_check, px_center.astype(int), 10, (0, 255, 0), -1)  # Draw a green circle
        # cv2.circle(dimentions_check, manual_px_center.astype(int), 10, (255, 0, 0), -1)  # Draw a green circle
        cv2.imwrite("calib_data/dimentions_check.png", dimentions_check)
        return rvecs, tvecs


if __name__ == "__main__":
    estimator = PoseEstimator()
    estimator.estimatePose(visualise = True)