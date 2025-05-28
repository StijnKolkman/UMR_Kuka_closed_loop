import cv2
import os
import csv
import numpy as np
import time

#tracker functions
def select_roi(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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

        # Add a short delay to give you time to inspect the thresholded image
        # time.sleep(3)  # Adjust the time as needed (0.5 sec for example)

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
            angle = rect[2]

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

        return box,roi,angle, center_x_, center_y_


def camera_to_box_distance(L_real_mm, L_pixels, focal_length_px):
    L_real_m = L_real_mm / 1000.0
    D_camera_box =(focal_length_px * L_real_m) / L_pixels
    return D_camera_box