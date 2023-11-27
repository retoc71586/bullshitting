#!/usr/bin/env python3

import cv2
import argparse
import numpy as np


class ArucoDetector:
    def __init__(self, aruco_dict=cv2.aruco.DICT_4X4_50, edge_length=0.011 * 6) -> None:
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.edge_length = edge_length

        self.camera_matrix = None
        self.dist_coeffs = None

        # Hardcoded intrinsics for the OAK-D ROS data
        self.K = np.array(
            [
                512.6467895507812,
                0.0,
                323.8077087402344,
                0.0,
                512.6467895507812,
                183.98281860351562,
                0.0,
                0.0,
                1.0,
            ]
        )
        self.D = np.array(
            [
                12.029667854309082,
                -115.97200775146484,
                0.002035579876974225,
                0.002628991613164544,
                373.9336242675781,
                11.838055610656738,
                -114.76853942871094,
                369.8811340332031,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        self.tag_corners_points = np.array(
            [
                [-edge_length / 2.0, edge_length / 2.0, 0.0],
                [edge_length / 2.0, edge_length / 2.0, 0.0],
                [edge_length / 2.0, -edge_length / 2.0, 0.0],
                [-edge_length / 2.0, -edge_length / 2.0, 0.0],
            ]
        )

    def get_markers_pose(
        self,
        video_file,
        visualise,
        marker_length=0.011 * 6,
        aruco_dict=cv2.aruco.DICT_4X4_50,
    ):
        """
        Given the path to an image returns tvces and rvecs for each detected marker

        @param video_file : path to the video file
        @return tvecs, rvecs : two lists of the translation and rotation vectors which represent the transformation from the marker to the camera
        """
        camera_matrix = self.K.reshape((3, 3))
        dist_coeffs = self.D

        tot_tvecs = []
        tot_rvecs = []

        # Open the video file
        video = cv2.VideoCapture(video_file)

        # Check if the video file was successfully opened
        if not video.isOpened():
            print("Error opening video file")
            exit()

        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(aruco_dict), parameters
        )

        # Read frames from the video
        while True:
            ret, frame = video.read()
            if not ret:
                break

            corners, ids, rejected = detector.detectMarkers(frame)
            print(corners)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.imshow("Frame", frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the video file and close windows
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program takes as input the path to an image and computes the pose (tvecs, rvecs) of all the detected aruco markers."
    )
    parser.add_argument(
        "-p",
        "--video_input_path",
        type=str,
        default="episode_0_success_oakd.mp4",
        help="Path to the video to process.",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize the detected markers in the video.",
    )
    args = parser.parse_args()
    assert (
        args.video_input_path is not None
    ), "Please provide a valid path to a video file"

    np.set_printoptions(suppress=True)

    detector = ArucoDetector()
    tvecs, rvecs = detector.get_markers_pose(
        video_file=args.video_input_path, visualise=args.visualize
    )
