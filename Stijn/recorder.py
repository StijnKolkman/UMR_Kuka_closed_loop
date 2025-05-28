import time
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np 
import os
import subprocess 
import csv

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from main import publish_pose, shutdown_publisher
import utils

cap_api = cv2.CAP_DSHOW  # Found to be the best API for using with logitech C920 in Windows. Other options are also possible

class ClosedLoopRecorder:
    def __init__(self, window):
        # Initialize variables for box, UMRs, reconstruction
        self.pose = None
        self.box_1_roi = None   
        self.box_2_roi = None
        self.UMR_1_roi = None
        self.UMR_2_roi = None
        self.UMR_1_angle = None
        self.UMR_2_angle = None
        self.UMR_1_center_y = None
        self.UMR_1_center_x = None
        self.UMR_2_center_y = None
        self.UMR_2_center_x = None
        self.UMR_1_bounding_box = None
        self.UMR_2_bounding_box = None
        self.reconstruction_boolean = False

        # Known box dimensions (mm)
        self.real_box_width_cam1_mm = 108
        self.real_box_height_cam1_mm = 56
        self.real_box_width_cam2_mm = 108
        self.real_box_height_cam2_mm = 32

        # Camera calibration params
        self.camera_matrix1 = np.array([
            [1397.9,   0, 953.6590],
            [   0, 1403.0, 555.1515],
            [   0,   0,   1]
        ], dtype=np.float64)
        self.dist_coeffs1 = np.array([0.1216, -0.1727, 0, 0, 0], dtype=np.float64)
        self.camera_matrix2 = np.array([
            [1397.9,   0, 953.6590],
            [   0, 1403.0, 555.1515],
            [   0,   0,   1]
        ], dtype=np.float64)
        self.dist_coeffs2 = np.array([0.1216, -0.1727, 0, 0, 0], dtype=np.float64)

        self.window = window
        self.window.title("Dual Camera Recorder")
        self.recording = False
        self.recorded_file_names = None 
        self.frames_cam1 = []
        self.frames_cam2 = []
        self.record_start_time = None

        # Set window size
        self.window.geometry("1920x700")

        # Cameras or videos
        #self.cap1 = cv2.VideoCapture(1)  # Uncomment if real cameras used
        #self.cap2 = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(r"/home/dev/ros2_ws/testVideos/Coated_nFin2_pitch2_0_4hz_cam1.avi")
        self.cap2 = cv2.VideoCapture(r"/home/dev/ros2_ws/testVideos/Coated_nFin2_pitch2_0_4hz_cam2.avi")

        # Setup 3 main frames side by side: cam1, cam2, plot
        self.cam1_frame = tk.Frame(self.window, bd=2, relief="sunken")
        self.cam2_frame = tk.Frame(self.window, bd=2, relief="sunken")
        self.plot_frame = tk.Frame(self.window, bd=2, relief="sunken")
        self.controls_frame = tk.Frame(self.window, bd=2, relief="ridge")

        # Grid layout for main window
        self.cam1_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.cam2_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.plot_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.controls_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Configure grid weights for resizing
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_columnconfigure(2, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        # Camera video labels
        self.video_label1 = tk.Label(self.cam1_frame)
        self.video_label1.pack(fill="both", expand=True)
        self.video_label2 = tk.Label(self.cam2_frame)
        self.video_label2.pack(fill="both", expand=True)

        # Matplotlib figure and canvas in plot_frame
        self.fig = plt.Figure(figsize=(6, 5))
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # === Controls in controls_frame ===
        # Left column for labels, entries, sliders, buttons

        # File name label and entry
        self.filename_label = tk.Label(self.controls_frame, text="File name:")
        self.filename_label.grid(row=0, column=0, sticky="w", padx=5)
        self.filename_entry = tk.Entry(self.controls_frame)
        self.filename_entry.insert(0, "Recording")
        self.filename_entry.grid(row=0, column=1, sticky="ew", padx=5)

        # Focus sliders and labels for cam1
        self.focus_label1 = tk.Label(self.controls_frame, text="Focus Camera 1")
        self.focus_label1.grid(row=1, column=0, sticky="w", padx=5)
        self.focus_slider1 = ttk.Scale(self.controls_frame, from_=0, to=255, orient='horizontal', length=200, command=self.set_focus1)
        self.focus_slider1.set(58)
        self.focus_slider1.grid(row=1, column=1, sticky="ew", padx=5)
        self.focus_value_label1 = tk.Label(self.controls_frame, text=f"Value: {self.focus_slider1.get():.2f}")
        self.focus_value_label1.grid(row=1, column=2, sticky="w", padx=5)

        # Focus sliders and labels for cam2
        self.focus_label2 = tk.Label(self.controls_frame, text="Focus Camera 2")
        self.focus_label2.grid(row=2, column=0, sticky="w", padx=5)
        self.focus_slider2 = ttk.Scale(self.controls_frame, from_=0, to=255, orient='horizontal', length=200, command=self.set_focus2)
        self.focus_slider2.set(91)
        self.focus_slider2.grid(row=2, column=1, sticky="ew", padx=5)
        self.focus_value_label2 = tk.Label(self.controls_frame, text=f"Value: {self.focus_slider2.get():.2f}")
        self.focus_value_label2.grid(row=2, column=2, sticky="w", padx=5)

        # Recording button
        self.record_button = tk.Button(self.controls_frame, text="Start recording", command=self.toggle_recording, bg="red", fg="white")
        self.record_button.grid(row=3, column=0, sticky="ew", padx=5, pady=10)

        # Calibration buttons for cams 1 and 2
        self.calibration_button1 = tk.Button(self.controls_frame, text="Calibrate cam 1", command=self.toggle_calibration_cam1, bg="blue", fg="white")
        self.calibration_button1.grid(row=3, column=1, sticky="ew", padx=5, pady=10)

        self.calibration_button2 = tk.Button(self.controls_frame, text="Calibrate cam 2", command=self.toggle_calibration_cam2, bg="blue", fg="white")
        self.calibration_button2.grid(row=3, column=2, sticky="ew", padx=5, pady=10)

        # Trajectory reconstruction button
        self.trajectory_reconstructor = tk.Button(self.controls_frame, text="Start trajectory reconstruction", command=self.start_trajectory_reconstruction, bg="blue", fg="white")
        self.trajectory_reconstructor.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=10)

        # Label for recorded files info
        self.recorded_files_label = tk.Label(self.controls_frame, text="", fg="blue")
        self.recorded_files_label.grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=10)

        # Make columns 1 and 2 expandable in controls_frame
        self.controls_frame.grid_columnconfigure(1, weight=1)
        self.controls_frame.grid_columnconfigure(2, weight=1)

        # Initiate timestamps array
        self.timestamps = []

        # Start video update loop
        self.update_frame()

        # Cleanup on close
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def set_focus1(self, val):
        # Update the focus on camera1
        focus_value = float(val)
        self.cap1.set(cv2.CAP_PROP_FOCUS, focus_value)
        if hasattr(self, 'focus_value_label1'):  # Ensure the label exists before updating
            # Update the label
            self.focus_value_label1.config(text=f"Focus Camera 1 Value: {focus_value:.2f}")

    def set_focus2(self, val):
        # Update the focus on camera2
        focus_value = float(val)
        self.cap2.set(cv2.CAP_PROP_FOCUS, focus_value)
        if hasattr(self, 'focus_value_label2'):  # Ensure the label exists before updating
            # Update the label
            self.focus_value_label2.config(text=f"Focus Camera 2 Value: {focus_value:.2f}")

    def set_recording_done_callback(self, callback):
        # needed to send to  main that the recording is done and the tracker should start
        self.recording_done_callback = callback

    def toggle_recording(self):
        self.recording = not self.recording
        filename = self.filename_entry.get().strip() or "recording"
        output_dir = os.path.join(os.getcwd(), filename)
        os.makedirs(output_dir, exist_ok=True)

        cam1_filename = os.path.join(output_dir, f"{filename}_cam1.avi")
        cam2_filename = os.path.join(output_dir, f"{filename}_cam2.avi")

        if self.recording:
            # Create video writer
            self.frames_cam1 = []
            self.frames_cam2 = []
            self.record_start_time = time.time()
            self.record_button.config(text="Stop recording", bg="gray")
            self.recorded_files_label.config(text="Recording in progress...")
            print(f"Started recording: {cam1_filename} & {cam2_filename}")
            
            self.out1 = cv2.VideoWriter(cam1_filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
            self.out2 = cv2.VideoWriter(cam2_filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))

        else:
            # Stop recording and calculate FPS
            duration = time.time() - self.record_start_time
            fps_value = len(self.frames_cam1) / duration if duration > 0 else 30.0
            print(f"Duration: {duration:.2f}s — FPS: {fps_value:.2f}")

            # Adjust the FPS value of the video writers after recording
            self.out1.set(cv2.CAP_PROP_FPS, fps_value)
            self.out2.set(cv2.CAP_PROP_FPS, fps_value)
            self.out1.release()
            self.out2.release()
            #self.fix_video_fps_inplace(cam1_filename, fps_value)
            #self.fix_video_fps_inplace(cam2_filename, fps_value)
            #print(f'Fixed video 1 and 2 fps to {fps_value}')

            self.record_button.config(text="Start recording", bg="red")
            files_text = f"Recorded files:\n{cam1_filename}\n{cam2_filename}"
            self.recorded_files_label.config(text=files_text)
            print("Recording done and saved")
            self.recorded_file_names = (cam1_filename, cam2_filename)  # Store filenames

            # Save timestamps to CSV
            timestamp_filename = os.path.join(output_dir, f"{filename}_timestamps.csv")
            with open(timestamp_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Timestamp (s)"])
                for i, ts in enumerate(self.timestamps):
                    writer.writerow([i, ts])

            print(f"[INFO] Timestamps saved to {timestamp_filename}")

            # Call the callback when recording is done, but only if files are recorded
            if hasattr(self, 'recording_done_callback') and self.recorded_file_names:
                self.recording_done_callback()  # Notify that recording is done

    def toggle_calibration_cam1(self):
        self.box_1_roi = utils.select_roi(self.cap1)
        time.sleep(2)
        self.UMR_1_roi = utils.select_roi(self.cap1)
        print('calibration done...')
        
        # Update the button text and color to indicate calibration is done
        self.calibration_button1.config(bg="blue", fg="white", text="Calibrate cam 1")

    def toggle_calibration_cam2(self):
        self.box_2_roi = utils.select_roi(self.cap2)
        time.sleep(2)
        self.UMR_2_roi = utils.select_roi(self.cap2)
        print('calibration done...')

        # Update the button text and color to indicate calibration is done
        self.calibration_button2.config(bg="blue", fg="white", text="Calibrate cam 2")

    def start_trajectory_reconstruction(self):

        self.reconstruction_boolean = True
        # Get the focal lengths and the optical centers
        self.fx_cam1 = self.camera_matrix1[0,0]
        self.fy_cam1 = self.camera_matrix1[1,1]
        self.fx_cam2 = self.camera_matrix2[0,0]
        self.fy_cam2 = self.camera_matrix2[1,1]
        self.cx_cam1 = self.camera_matrix1[0, 2]
        self.cy_cam1 = self.camera_matrix1[1, 2]
        self.cy_cam2 = self.camera_matrix2[1, 2]

        # Initialize the distances and 3d location
        self.Z1 = []
        self.Z2 = []
        self.X_3d = []
        self.Y_3d = []
        self.Z_3d = []

        #trajectory reconstruction variables
        self.box_1_x, self.box_1_y, self.box_1_width_px, self.box_1_height_px = self.box_1_roi
        self.box_2_x, self.box_2_y, self.box_2_width_px, self.box_2_height_px = self.box_2_roi

        x, y, w, h = self.box_1_roi
        self.umr_1_center_x = x + w // 2
        self.umr_1_center_y = y + h // 2

        x, y, w, h = self.box_2_roi
        self.umr_2_center_x = x + w // 2
        self.umr_2_center_y = y + h // 2

        self.mm_per_pixel_cam_1_x = self.real_box_width_cam1_mm / self.box_1_width_px 
        self.mm_per_pixel_cam_1_y = self.real_box_height_cam1_mm / self.box_1_height_px
        self.mm_per_pixel_cam_2_x = self.real_box_width_cam2_mm / self.box_2_width_px 
        self.mm_per_pixel_cam_2_y = self.real_box_height_cam2_mm / self.box_2_height_px

        self.cam1_to_box_distance = utils.camera_to_box_distance(self.real_box_width_cam1_mm,self.box_1_width_px,self.fx_cam1)
        self.cam2_to_box_distance = utils.camera_to_box_distance(self.real_box_width_cam2_mm,self.box_2_width_px,self.fx_cam2)

        bottom_box_cam1 = self.box_1_y+self.box_1_height_px
        bottom_box_cam2 = self.box_2_y+self.box_2_height_px

        self.initial_y = (bottom_box_cam1-self.umr_1_center_y)*(self.mm_per_pixel_cam_1_y/1000)
        self.initial_z = (bottom_box_cam2-self.umr_2_center_y)*(self.mm_per_pixel_cam_2_y/1000)

        # Calculate the initial Z position based on the box height and camera distance
        self.Z1_initial = self.cam1_to_box_distance + self.initial_z # Distance between object and camera1 at time=0
        self.Z2_initial = self.cam2_to_box_distance + self.initial_y

        self.Z1.append(self.Z1_initial)  # Initial depth for camera 1
        self.Z2.append(self.Z2_initial)

    def update_trajectory_plot(self):
        self.ax3d.clear()

        # Shift data to box corner
        x_box_px, y_box_px = self.box_1_x, self.box_1_y  # pixel coords of box corner
        Z_box = self.cam1_to_box_distance  # depth in meters
        X_box = (x_box_px - self.cx_cam1) * Z_box / self.fx_cam1
        Y_box = (y_box_px - self.cy_cam1) * Z_box / self.fy_cam1

        # Your 3D trajectory data
        X = (np.array(self.X_3d)-X_box)*1000 if hasattr(self, 'X_3d') else np.array([])
        Y = (np.array(self.Y_3d)-Y_box)*1000 if hasattr(self, 'Y_3d') else np.array([])
        Z = (np.array(self.Z_3d))*1000 if hasattr(self, 'Z_3d') else np.array([])

        # Print trajectory points
        print("Plotting 3D trajectory points:")
        for x, y, z in zip(X, Y, Z):
            print(f"X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm")

        # Plot trajectory if available
        if X.size > 0:
            self.ax3d.plot(X, Y, Z, color='blue', linewidth=2, label='Trajectory')

        # Draw 3D box (rectangular prism)
        w = self.real_box_width_cam1_mm      # width in mm
        h = self.real_box_height_cam1_mm     # height in mm (camera 1 view)
        d = self.real_box_height_cam2_mm     # depth in mm (camera 2 view)

        # Define box corners (starting at origin)
        corners = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0],
            [0, 0, d],
            [w, 0, d],
            [w, h, d],
            [0, h, d]
        ])

        # List of edges (pairs of corner indices)
        edges = [
            (0,1), (1,2), (2,3), (3,0),   # bottom face
            (4,5), (5,6), (6,7), (7,4),   # top face
            (0,4), (1,5), (2,6), (3,7)    # vertical edges
        ]

        # Plot edges
        for start, end in edges:
            xs = [corners[start,0], corners[end,0]]
            ys = [corners[start,1], corners[end,1]]
            zs = [corners[start,2], corners[end,2]]
            self.ax3d.plot(xs, ys, zs, color='red', linestyle='--', linewidth=1, label='Box' if start==0 and end==1 else "")

        self.ax3d.set_title("3D Trajectory with Calibration Box")
        self.ax3d.set_xlabel("X (mm)")
        self.ax3d.set_ylabel("Y (mm)")
        self.ax3d.set_zlabel("Z (mm)")

        # Equal aspect ratio for cubic box
        max_range = np.array([w, h, d]).max() / 2.0
        mid_x = w / 2.0
        mid_y = h / 2.0
        mid_z = d / 2.0

        self.ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax3d.view_init(elev=30, azim=45)

        self.ax3d.legend()
        self.canvas.draw()

    def send_pose(self):
        """
        Publish the pose to the ROS topic.
        """
        if self.pose is not None:
            publish_pose(self.pose)
            print(f"Published pose: {self.pose}")
        else:
            print("No pose to publish.")

    def update_trajectory(self):
        #undistort the UMRs
        point_cam1 = np.array([[[self.UMR_1_center_x, self.UMR_1_center_y]]], dtype=np.float32)  # shape (1,1,2)
        undistorted_point_cam1 = cv2.undistortPoints(point_cam1, self.camera_matrix1, self.dist_coeffs1, P=self.camera_matrix1)
        self.UMR_1_center_x, self.UMR_1_center_y = undistorted_point_cam1[0,0]

        point_cam2 = np.array([[[self.UMR_2_center_x, self.UMR_2_center_y]]], dtype=np.float32)
        undistorted_point_cam2 = cv2.undistortPoints(point_cam2, self.camera_matrix2, self.dist_coeffs2, P=self.camera_matrix2)
        self.UMR_2_center_x, self.UMR_2_center_y = undistorted_point_cam2[0,0]


        # Loop to calculate the 3d position and update the distances per timestep
        X_3d_next = (self.UMR_1_center_x - self.cx_cam1)*self.Z1[-1]/self.fx_cam1
        Y_3d_next = (self.UMR_1_center_y - self.cy_cam1)*self.Z1[-1]/self.fy_cam1
        Z_3d_next = -(self.UMR_2_center_y - self.cy_cam2)*self.Z2[-1]/self.fy_cam2

        # Append the new 3D coordinates to the lists
        self.X_3d.append(X_3d_next)
        self.Y_3d.append(Y_3d_next)
        self.Z_3d.append(Z_3d_next)
        # Adjust future depth estimates based on movement in Y and Z,

        # this is assuming smooth motion and small displacements
        Z1_next = self.Z1[0]+(self.Z_3d[-1] - self.Z_3d[0])
        Z2_next = self.Z2[0]-(self.Y_3d[-1] - self.Y_3d[0])
        self.Z1.append(Z1_next)
        self.Z2.append(Z2_next)

        # Make the first position 0,0,0
        #X_3d -= X_3d[0]
        #Y_3d -= Y_3d[0]
        #Z_3d -= Z_3d[0]

    def update_frame(self):
        
        # Check if the window is still open before updating
        if not self.window.winfo_exists():
            return
        
        #Read the frames
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if self.recording and ret1 and ret2:
            # Save frames
            self.frames_cam1.append(frame1.copy())
            self.frames_cam2.append(frame2.copy())
            self.out1.write(frame1)
            self.out2.write(frame2)
            timestamp = time.time() - self.record_start_time
            self.timestamps.append(timestamp)

        if ret1:
            # Update the GUI with the frame --> first the frame is resized to fit in the GUI
            frame1_display = frame1.copy()
            if self.UMR_1_roi is not None:
                x, y, w, h = [int(v) for v in self.box_1_roi]
                cv2.rectangle(frame1_display, (x,y), (x+w,y+h), (255,0,0), 4)
                # Update the tracker 
                self.UMR_1_bounding_box, self.UMR_1_roi,self.UMR_1_angle, self.UMR_1_center_x, self.UMR_1_center_y = utils.update_roi_center(frame1, self.UMR_1_roi)
                x, y, w, h = [int(v) for v in self.UMR_1_roi]
                #cv2.rectangle(frame1_display, (x,y), (x+w,y+h), (0,0,255), 4) #This can draw the ROI around the umr, but it is not needed
                # Draw the bounding box around the umr 
                cv2.drawContours(frame1_display, [self.UMR_1_bounding_box + (x, y)], 0, (0, 0, 255), 2)

            frame1_resized = cv2.resize(frame1_display, (576, 324), interpolation=cv2.INTER_LINEAR)    
            frame_rgb1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
            img1 = ImageTk.PhotoImage(Image.fromarray(frame_rgb1))
            self.video_label1.imgtk = img1
            self.video_label1.config(image=img1)


        if ret2:
            # Update the GUI with the frame --> first the frame is resized to fit in the GUI 
            frame2_display = frame2.copy()
            if self.UMR_2_roi is not None:
                x, y, w, h = [int(v) for v in self.box_2_roi]
                cv2.rectangle(frame2_display, (x,y), (x+w,y+h), (255,0,0), 4)
                # Update the tracker 
                self.UMR_2_bounding_box,self.UMR_2_roi,self.UMR_2_angle, self.UMR_2_center_x, self.UMR_2_center_y = utils.update_roi_center(frame2, self.UMR_2_roi)
                x, y, w, h = [int(v) for v in self.UMR_2_roi]
                #cv2.rectangle(frame2_display, (x,y), (x+w,y+h), (0,0,255), 4) #This can draw the ROI around the umr, but it is not needed
                # Draw the bounding box around the umr
                cv2.drawContours(frame2_display, [self.UMR_2_bounding_box + (x, y)], 0, (0, 0, 255), 2)    
            frame2_resized = cv2.resize(frame2_display, (576, 324), interpolation=cv2.INTER_LINEAR)    
            frame_rgb2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
            img2 = ImageTk.PhotoImage(Image.fromarray(frame_rgb2))
            self.video_label2.imgtk = img2
            self.video_label2.config(image=img2)

        # Update the reconstructed pose based on the UMRs and box positions
        if self.reconstruction_boolean is True:
            self.update_trajectory()
            self.update_trajectory_plot()

        self.window.after(10, self.update_frame)        

    def on_closing(self):
        self.cap1.release()
        self.cap2.release()
        self.window.destroy()
        shutdown_publisher()

# Used if the recorder class is called seperately
if __name__ == "__main__":
    root = tk.Tk()
    app = ClosedLoopRecorder(root)
    root.mainloop()

    #TODO fix the error met gui selection --> misschien werkt het op linux wel goed 
    #TODO fix the angle calculation for the UMR's
    #TODO: use the correct compensation for the corner and optical center
    #self.pose = (3.14,0.0,0.0,0.0,0.0,0.0)
    #self.send_pose()
