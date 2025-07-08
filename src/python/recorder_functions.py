import math
import numpy as np
import os
import csv
import cv2
import time
from scipy.spatial.transform import Rotation as R


def generate_relative_linear_trajectory_3d(X0,Y0,Z0,length_m=0.1,num_points=50,direction_rad=0.0):
    # get initial world‐XYZ (already in meters)

    # compute endpoint along XY‐direction
    dx = length_m * math.cos(direction_rad)
    dy = length_m * math.sin(direction_rad)
    X1 = X0 + dx
    Y1 = Y0 + dy
    Z0 = Z0 

    # linearly interpolate X, Y; keep Z constant
    x_vals = np.linspace(X0, X1, num_points)
    y_vals = np.linspace(Y0, Y1, num_points)
    z_vals = np.full(num_points, Z0, dtype=np.float64)
    theta_vals = np.full(num_points, direction_rad, dtype=np.float64)

    # Stack into (num_points × 4): [x_m, y_m, z_m, θ_rad]
    trajectory_3d = np.vstack((x_vals, y_vals, z_vals, theta_vals)).T
    return trajectory_3d

def generate_curved_trajectory_3d(X0,Y0,Z0,radius_m=0.1, arc_angle_rad=math.pi/2,num_points=50,direction_rad=0.0, turn_left=True):
    psi0 = direction_rad
    sweep = arc_angle_rad if turn_left else -arc_angle_rad

    # center of the turning circle
    if turn_left:
        Xc = X0 - radius_m * math.sin(psi0)
        Yc = Y0 + radius_m * math.cos(psi0)
    else:
        Xc = X0 + radius_m * math.sin(psi0)
        Yc = Y0 - radius_m * math.cos(psi0)

    phi = np.linspace(0, sweep, num_points)

    Xs = Xc + radius_m * np.sin(psi0 + phi)
    Ys = Yc - radius_m * np.cos(psi0 + phi)

    # tangent heading at each waypoint
    if turn_left:
        theta = psi0 + phi + (math.pi / 2)
    else:
        theta = psi0 + phi - (math.pi / 2)

    # wrap into [0, 2π)
    theta = theta % (2 * math.pi)

    Zs = np.full(num_points, Z0, dtype=np.float64)

    # Stack into (num_points × 4): [x_m, y_m, z_m, θ_rad]
    trajectory_3d = np.vstack((Xs, Ys, Zs, theta)).T
    return trajectory_3d

def save_trajectory_to_csv(trajectory_3d,filename_entry):
    # Validate trajectory shape
    if not isinstance(trajectory_3d, np.ndarray) or trajectory_3d.ndim != 2 or trajectory_3d.shape[1] != 4:
        raise ValueError(f"Invalid trajectory shape: expected (N, 4), got {trajectory_3d.shape}")

    filename = filename_entry.get().strip() or "recording"
    output_dir = os.path.join(os.getcwd(), filename)
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    output_file_path = os.path.join(output_dir, f"{filename}_reference_trajectory.csv")

    # Save trajectory to CSV
    header = ['x_m', 'y_m', 'z_m', 'theta_rad']
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(trajectory_3d)

    print(f"Trajectory saved to: {output_file_path}")

def camera_to_box_distance(L_real_mm, L_pixels, focal_length_px):
    L_real_m = L_real_mm / 1000.0
    D_camera_box =(focal_length_px * L_real_m) / L_pixels
    return D_camera_box

def select_roi(cap):
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from the video.")
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError("Video reading error.")

    # Resize frame for a consistent ROI selection window size
    max_width = 1280
    max_height = 720
    height, width = frame.shape[:2]
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # scale down only, never upscale

    frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    time.sleep(1)   # Allow time for the frame to be displayed
    roi_scaled = cv2.selectROI("Select the ROI", frame_resized, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select the ROI")
    cv2.waitKey(100)  # small delay to allow window to close properly

    # Scale ROI back to original frame size
    x, y, w, h = roi_scaled
    roi = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

    return roi

def update_roi_center(frame, roi):

        # Crop the ROI from the frame
        x, y, w, h = [int(v) for v in roi]

        # Ensure that the ROI coordinates are within the frame's dimensions
        height, width = frame.shape[:2]
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y

        roi_frame = frame[y:y+h, x:x+w]

        # Convert to grayscale and apply Otsu's thresholding
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Debug: Show the thresholded image
        #cv2.imshow("Thresholded Image", threshold)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            #x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(largest_contour)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Compute the object's center (account for ROI offset)
            center_x_, center_y_ = rect[0]
            center_x_ += x
            center_y_ += y
            center_x_, center_y_ = int(center_x_), int(center_y_)
            #angle = rect[2]

            ellipse = cv2.fitEllipse(largest_contour) # --> probably better option to get good angle between -90 and 90
            angle = math.radians(ellipse[2]-90)  # in radians and make horizontal zero

            # Update the ROI center based on the object's new center
            # Keep the original size (w, h) but adjust its position
            new_x = center_x_ - w // 2
            new_y = center_y_ - h // 2

            # Ensure the new ROI is within the bounds of the frame
            new_x = max(new_x, 0)
            new_y = max(new_y, 0)

            # Ensure the ROI does not go out of bounds
            if new_x + w > width:
                new_x = width - w
            if new_y + h > height:
                new_y = height - h

            # Update the ROI
            roi = (new_x, new_y, w, h)

            # Draw the contour and center on the frame for debugging
            #cv2.circle(frame, (center_x_, center_y_), 5, (0, 0, 255), -1)
            # Adjust drawn contours by the top-left ROI offset (x, y)
            #cv2.drawContours(frame, [box + (x, y)], 0, (0, 0, 255), 2)
            #cv2.putText(frame, f"Orientation: {angle:.2f} deg", (x, y - 10),
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            print("No contours found.")

        return box, roi, angle, center_x_, center_y_

def find_nearest_trajectory_point(trajectory_3d, current_x, current_y):
    trajectory_xy = trajectory_3d[:, :2]
    current_pos = np.array([current_x, current_y])
    distances = np.linalg.norm(trajectory_xy - current_pos, axis=1)
    index = np.argmin(distances)
    return index

def transform_pose(R_BoxToKuka, old_pos, old_rot):

    # Transform position
    new_pos = R_BoxToKuka @ np.array(old_pos)

    # Convert original Euler angles to rotation matrix
    rot_matrix = R.from_euler('xyz', old_rot).as_matrix()

    # Transform rotation matrix: R_new = T * R_old * T.T
    rot_transformed = R_BoxToKuka @ rot_matrix @ R_BoxToKuka.T

    # Convert back to Euler angles
    new_rot = -R.from_matrix(rot_transformed).as_euler('xyz') #I think the rotations go in the wrong direction, thats why a minus was added (no clue where goes wrong. Maybe my angle definition in the cameras is wrong)

    return new_pos, new_rot