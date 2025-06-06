import time
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np 
import os
import csv
import math
import rclpy

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kuka_python_node.kuka_node import start_kuka_node

class ClosedLoopRecorder:
    def __init__(self, window):
        self.kuka_python = start_kuka_node()

        # ----------------------------
        # 1) Initialize state & ROS node
        # ----------------------------

        # Filtered angles
        self.angle_1_filtered = None
        self.angle_2_filtered = None
        self.alpha = 0.2
        self.i = 0

        # Controller state
        self.controller_boolean = False
        self.motor_velocity_rad = 0.0

        # Reconstruction / ROI state
        self.pose = None
        self.box_1_roi = None
        self.box_2_roi = None
        self.UMR_1_roi = None
        self.UMR_2_roi = None
        self.UMR_1_center_x = None
        self.UMR_1_center_y = None
        self.UMR_2_center_x = None
        self.UMR_2_center_y = None
        self.UMR_1_bounding_box = None
        self.UMR_2_bounding_box = None
        self.kuka_start_pos = None
        self.kuka_start_rot = None
        self.reconstruction_boolean = False

        # Known box dimensions (mm)
        self.real_box_width_cam1_mm = 108
        self.real_box_height_cam1_mm = 56
        self.real_box_width_cam2_mm = 108
        self.real_box_height_cam2_mm = 32

        # Camera calibration parameters
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

        # ----------------------------
        # 2) Configure main window
        # ----------------------------
        self.window = window
        self.window.title("Dual Camera Recorder")
        self.window.geometry("1024x700")
        self.window.minsize(800, 600)

        # Use a simple ttk theme for consistency
        style = ttk.Style(self.window)
        style.theme_use('default')
        style.configure('TLabel', font=('Arial', 10))
        style.configure('TButton', padding=6, font=('Arial', 10))
        style.configure('TEntry', padding=4)
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabelframe', background='#f0f0f0', font=('Arial', 10, 'bold'))
        style.configure('TLabelframe.Label', font=('Arial', 10, 'bold'))
        style.configure('Calibrated.TButton', background='green', foreground='white')
        style.configure('ControllerOn.TButton', background='green', foreground='white')
        style.configure('ControllerOff.TButton', background='red', foreground='white')
        style.configure('CalibrateKuka.TButton',background='red',foreground='white')
        style.map('CalibrateKuka.TButton',background=[('active', '#005f00')],foreground=[('active', 'white')])
        style.configure('CalibrateKukaDone.TButton',background='#007f00',foreground='white')
        style.map('CalibrateKukaDone.TButton',background=[('active', '#005f00')],foreground=[('active', 'white')])
        style.configure('RecordingOff.TButton', background='red', foreground='white')
        style.configure('RecordingOn.TButton',  background='gray', foreground='white')

        # Make columns 0,1,2 expand equally; row 0 expands vertically
        self.window.grid_columnconfigure((0,1,2), weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        # ----------------------------
        # 3) Open video captures
        # ----------------------------
        # If using actual cameras: cv2.VideoCapture(0), cv2.VideoCapture(1)
        self.cap1 = cv2.VideoCapture(r"/home/dev/ros2_ws/videos/Coated_pitch1_0_4hz_v2_cam1.avi")
        self.cap2 = cv2.VideoCapture(r"/home/dev/ros2_ws/videos/Coated_pitch1_0_4hz_v2_cam2.avi")
        self.cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # ----------------------------
        # 4) Set up main frames
        # ----------------------------
        # Left: Camera 1 preview; Middle: Camera 2 preview; Right: 3D plot
        self.cam1_frame = ttk.Frame(self.window, borderwidth=2, relief="sunken")
        self.cam2_frame = ttk.Frame(self.window, borderwidth=2, relief="sunken")
        self.plot_frame = ttk.Frame(self.window, borderwidth=2, relief="sunken")
        self.controls_frame = ttk.Frame(self.window, borderwidth=2, relief="ridge")

        self.cam1_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.cam2_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.plot_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.controls_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Make controls_frame’s columns 0,1,2 expand equally
        self.controls_frame.grid_columnconfigure((0,1,2), weight=1)

        # ----------------------------
        # 5) Camera preview labels
        # ----------------------------
        self.video_label1 = ttk.Label(self.cam1_frame)
        self.video_label1.pack(fill="both", expand=True)
        self.video_label2 = ttk.Label(self.cam2_frame)
        self.video_label2.pack(fill="both", expand=True)

        # ----------------------------
        # 6) Matplotlib 3D plot + toolbar
        # ----------------------------
        self.fig = plt.Figure(figsize=(6, 5))
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()

        # ----------------------------
        # 7) Build sub-panels in controls_frame
        # ----------------------------
        # 7.1 Recording group
        self.rec_frame = ttk.Labelframe(self.controls_frame, text="Recording", labelanchor="n", padding=(10,8))
        self.rec_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.rec_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.rec_frame, text="File name:").grid(row=0, column=0, sticky="w", padx=5, pady=(0,4))
        self.filename_entry = ttk.Entry(self.rec_frame)
        self.filename_entry.insert(0, "Recording")
        self.filename_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=(0,4))
        self.record_button = ttk.Button(self.rec_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4,0))

        # 7.2 Calibration group
        self.cal_frame = ttk.Labelframe(self.controls_frame, text="Calibration", labelanchor="n", padding=(10,8))
        self.cal_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.cal_frame.grid_columnconfigure(0, weight=1)

        self.calibration_button1 = ttk.Button(self.cal_frame, text="Calibrate Cam 1", command=self.toggle_calibration_cam1,style='ControllerOff.TButton')
        self.calibration_button1.grid(row=0, column=0, sticky="ew", pady=2)
        self.calibration_button2 = ttk.Button(self.cal_frame, text="Calibrate Cam 2", command=self.toggle_calibration_cam2,style='ControllerOff.TButton')
        self.calibration_button2.grid(row=1, column=0, sticky="ew", pady=2)

        self.calibrate_kuka_button = ttk.Button(self.cal_frame, text="Calibrate Kuka",command=self.calibrate_kuka,style='CalibrateKuka.TButton')
        self.calibrate_kuka_button.grid(row=2, column=0, sticky="ew", pady=4)


        self.trajectory_reconstructor = ttk.Button(self.cal_frame, text="Start 3D Reconstruction", command=self.start_trajectory_reconstruction)
        self.trajectory_reconstructor.grid(row=3, column=0, sticky="ew", pady=2)

        self.recorded_files_label = ttk.Label(self.cal_frame, text="", foreground="blue")
        self.recorded_files_label.grid(row=4, column=0, sticky="w", pady=(4,0))

        # 7.3 Controller group
        self.ctrl_frame = ttk.Labelframe(self.controls_frame, text="Controller", labelanchor="n", padding=(10,8))
        self.ctrl_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.ctrl_frame.grid_columnconfigure(1, weight=1)

        # Velocity input
        ttk.Label(self.ctrl_frame, text="Velocity (Hz):").grid(row=0, column=0, sticky="w", padx=5, pady=(0,4))
        self.velocity_entry = ttk.Entry(self.ctrl_frame)
        self.velocity_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=(0,4))
        self.set_velocity_button = ttk.Button(self.ctrl_frame, text="Set", command=self.set_new_velocity)
        self.set_velocity_button.grid(row=0, column=2, sticky="ew", padx=5, pady=(0,4))

        # Controller toggle
        self.controller_button = ttk.Button(self.ctrl_frame, text="Controller: OFF", command=self.toggle_controller,style='TButton')
        self.controller_button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=2)

        # Angle displays
        self.angle1_label = ttk.Label(self.ctrl_frame, text="Angle 1: ---°")
        self.angle1_label.grid(row=3, column=0, sticky="w", padx=5, pady=(6,0))
        self.angle2_label = ttk.Label(self.ctrl_frame, text="Angle 2: ---°")
        self.angle2_label.grid(row=3, column=1, sticky="w", padx=5, pady=(6,0))

        # Focus sliders (beneath controller group for compactness)
        # Cam 1 focus
        self.focus_label1 = ttk.Label(self.ctrl_frame, text="Focus Cam 1:")
        self.focus_label1.grid(row=4, column=0, sticky="w", padx=5, pady=(6,0))
        self.focus_slider1 = ttk.Scale(self.ctrl_frame, from_=0, to=255, orient='horizontal', command=self.set_focus1)
        self.focus_slider1.set(58)
        self.focus_slider1.grid(row=4, column=1, columnspan=2, sticky="ew", padx=5, pady=(6,0))

        # Cam 2 focus
        self.focus_label2 = ttk.Label(self.ctrl_frame, text="Focus Cam 2:")
        self.focus_label2.grid(row=5, column=0, sticky="w", padx=5, pady=(4,0))
        self.focus_slider2 = ttk.Scale(self.ctrl_frame, from_=0, to=255, orient='horizontal', command=self.set_focus2)
        self.focus_slider2.set(91)
        self.focus_slider2.grid(row=5, column=1, columnspan=2, sticky="ew", padx=5, pady=(4,0))

        # ----------------------------
        # 8) Recording variables & start loop
        # ----------------------------
        self.recording = False
        self.recorded_file_names = None
        self.frames_cam1 = []
        self.frames_cam2 = []
        self.record_start_time = None
        self.timestamps = []

        # Start the update loop and handle window‐close
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def set_focus1(self, val):
        # Update the focus on camera1
        focus_value = float(val)
        self.cap1.set(cv2.CAP_PROP_FOCUS, focus_value)
        if hasattr(self, 'focus_value_label1'):  # Ensure the label exists before updating
            self.focus_value_label1.config(text=f"Focus Camera 1 Value: {focus_value:.2f}")

    def set_focus2(self, val):
        # Update the focus on camera2
        focus_value = float(val)
        self.cap2.set(cv2.CAP_PROP_FOCUS, focus_value)
        if hasattr(self, 'focus_value_label2'):  # Ensure the label exists before updating
            self.focus_value_label2.config(text=f"Focus Camera 2 Value: {focus_value:.2f}")

    def toggle_controller(self):
        self.controller_boolean = not self.controller_boolean
        #state = "ON" if self.controller_boolean else "OFF"
        if self.controller_boolean:
            self.controller_button.config(text=f"Controller: ON", style='ControllerOn.TButton')
            self.kuka_python.set_motor_speed(self.motor_velocity_rad)
        else:
            self.controller_button.config(text=f"Controller: OFF", style='ControllerOff.TButton')
            self.kuka_python.set_motor_speed(0.0)

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
            self.recorded_files_label.config(text="Recording in progress...")
            print(f"Started recording: {cam1_filename} & {cam2_filename}")
            self.record_button.configure(text="Stop recording", style='RecordingOn.TButton')
            self.out1 = cv2.VideoWriter(cam1_filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
            self.out2 = cv2.VideoWriter(cam2_filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))

        else:
            # Stop recording
            duration = time.time() - self.record_start_time
            fps_value = len(self.frames_cam1) / duration if duration > 0 else 30.0
            print(f"Duration: {duration:.2f}s — FPS: {fps_value:.2f}")

            # Adjust the FPS value of the video writers after recording
            #self.out1.set(cv2.CAP_PROP_FPS, fps_value)
            #self.out2.set(cv2.CAP_PROP_FPS, fps_value)
            self.out1.release()
            self.out2.release()

            files_text = f"Recorded files:\n{cam1_filename}\n{cam2_filename}"
            self.recorded_files_label.config(text=files_text)
            print("Recording done and saved")
            self.recorded_file_names = (cam1_filename, cam2_filename)  # Store filenames
            self.record_button.configure(text="Start recording", style='RecordingOff.TButton')

            # Save timestamps to CSV
            timestamp_filename = os.path.join(output_dir, f"{filename}_timestamps.csv")
            with open(timestamp_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Timestamp (s)"])
                for i, ts in enumerate(self.timestamps):
                    writer.writerow([i, ts])

            print(f"[INFO] Timestamps saved to {timestamp_filename}")

            # Initiate timestamps array for next recording
            self.timestamps = []

    def toggle_calibration_cam1(self):
        self.box_1_roi = self.select_roi(self.cap1)
        self.UMR_1_roi = self.select_roi(self.cap1)
        self.calibration_button1.config(text="Cam 1 Calibrated", style='Calibrated.TButton')

    def toggle_calibration_cam2(self):
        self.box_2_roi = self.select_roi(self.cap2)
        self.UMR_2_roi = self.select_roi(self.cap2)
        self.calibration_button2.config(text="Cam 2 Calibrated", style='Calibrated.TButton')

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

        x, y, w, h = self.UMR_1_roi
        self.umr_1_center_x = x + w // 2
        self.umr_1_center_y = y + h // 2

        x, y, w, h = self.UMR_2_roi
        self.umr_2_center_x = x + w // 2
        self.umr_2_center_y = y + h // 2

        self.mm_per_pixel_cam_1_x = self.real_box_width_cam1_mm / self.box_1_width_px 
        self.mm_per_pixel_cam_1_y = self.real_box_height_cam1_mm / self.box_1_height_px
        self.mm_per_pixel_cam_2_x = self.real_box_width_cam2_mm / self.box_2_width_px 
        self.mm_per_pixel_cam_2_y = self.real_box_height_cam2_mm / self.box_2_height_px

        self.cam1_to_box_distance = self.camera_to_box_distance(self.real_box_width_cam1_mm,self.box_1_width_px,self.fx_cam1)
        self.cam2_to_box_distance = self.camera_to_box_distance(self.real_box_width_cam2_mm,self.box_2_width_px,self.fx_cam2)

        bottom_box_cam1 = self.box_1_y+self.box_1_height_px
        bottom_box_cam2 = self.box_2_y+self.box_2_height_px

        self.initial_y = (bottom_box_cam1-self.umr_1_center_y)*(self.mm_per_pixel_cam_1_y/1000)
        self.initial_z = (bottom_box_cam2-self.umr_2_center_y)*(self.mm_per_pixel_cam_2_y/1000)

        # Calculate the initial Z position based on the box height and camera distance
        self.Z1_initial = self.cam1_to_box_distance + self.initial_z # Distance between object and camera1 at time=0
        self.Z2_initial = self.cam2_to_box_distance + self.initial_y

        self.Z1.append(self.Z1_initial)  # Initial depth for camera 1
        self.Z2.append(self.Z2_initial)

        # Now we assume that with the camera setup, the left bottom corner of camera 1 is the origin. In camera 2 this is the bottom right corner
        # calculate ofset to shift to origin
        x_box_1_px, y_box_1_px = self.box_1_x, self.box_1_y  # pixel coords of box 1 top corner
        origin_box1_x_px = x_box_1_px
        origin_box1_y_px = y_box_1_px + self.box_1_height_px
        Z_box = self.cam1_to_box_distance  # depth in meters
        self.X_shift = (origin_box1_x_px - self.cx_cam1) * Z_box / self.fx_cam1
        self.Y_shift = (origin_box1_y_px - self.cy_cam1) * Z_box / self.fy_cam1

        x_box_2_px, y_box_2_px = self.box_2_x, self.box_2_y  # pixel coords of box 2 corner
        origin_box2_y_px = y_box_2_px + self.box_2_height_px
        Z_box = self.cam2_to_box_distance  # depth in meters
        self.Z_shift = (origin_box2_y_px - self.cy_cam2) * Z_box / self.fy_cam2

        #now make the trajectory! linear or curved
        #self.trajectory_3d = self.generate_relative_linear_trajectory(length_mm=100,num_points=50,direction_deg=0.0)
        self.trajectory_3d = self.generate_curved_trajectory_3d(radius_mm=100,arc_angle_deg=90.0,num_points=50,direction_deg=0.0,turn_left=True)

    def update_trajectory_plot(self):

        self.ax3d.clear()

        #3D trajectory data to mm
        X = (np.array(self.X_3d))*1000
        Y = (np.array(self.Y_3d))*1000
        Z = (np.array(self.Z_3d))*1000 

        # Print trajectory points
        #print(f"X={X[-1]:.2f} mm, Y={Y[-1]:.2f} mm, Z={Z[-1]:.2f} mm")

        # Plot trajectory if available
        if X.size > 0:
            self.ax3d.plot(X, Y, Z, color='blue', linewidth=4, label='Trajectory')
        
        if hasattr(self, 'trajectory_3d'):
            planned = np.array(self.trajectory_3d)
            planned_X = planned[:, 0]
            planned_Y = planned[:, 1]
            planned_Z = planned[:, 2]
            self.ax3d.plot(planned_X, planned_Y, planned_Z, label="Planned",color='black',linestyle='--')

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

        # List of edge
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

        #self.ax3d.view_init(elev=90, azim=270) #--> view cam 1
        #self.ax3d.view_init(elev=30, azim=250) 

        self.ax3d.legend()
        self.canvas.draw()

    def camera_to_box_distance(self,L_real_mm, L_pixels, focal_length_px):
        L_real_m = L_real_mm / 1000.0
        D_camera_box =(focal_length_px * L_real_m) / L_pixels
        return D_camera_box

    def send_pose(self,pos,rot):
        """
        Publish the pose to the ROS topic. pos, rot should be in meter and radians
        """
        if pos is not None and rot is not None:
            #first convert to mm and degrees
            pos_mm = pos*1000
            rot_deg = np.degrees(rot)
            
            pose = np.concatenate((pos_mm,rot_deg)) 
            self.kuka_python.publish_pose(pose)
            print(f"Published pose: {pose}")
        else:
            print("No pose to publish.")
 
    def get_current_pose(self):
        """"
        get the current pose of the kuka in m and radians
        """
        kuka_pose = self.kuka_python.get_position()
        print(f"kuka pose: {kuka_pose}")
        pos = np.array(kuka_pose[0:3], dtype=float)
        rot = np.array(kuka_pose[3:6], dtype=float)
        return pos, rot

    def update_trajectory(self):

        #undistort the UMRs
        point_cam1 = np.array([[[self.UMR_1_center_x, self.UMR_1_center_y]]], dtype=np.float32)  # shape (1,1,2)
        undistorted_point_cam1 = cv2.undistortPoints(point_cam1, self.camera_matrix1, self.dist_coeffs1, P=self.camera_matrix1)
        self.UMR_1_center_x, self.UMR_1_center_y = undistorted_point_cam1[0,0]

        point_cam2 = np.array([[[self.UMR_2_center_x, self.UMR_2_center_y]]], dtype=np.float32)
        undistorted_point_cam2 = cv2.undistortPoints(point_cam2, self.camera_matrix2, self.dist_coeffs2, P=self.camera_matrix2)
        self.UMR_2_center_x, self.UMR_2_center_y = undistorted_point_cam2[0,0]

        # Calculate position relative to focal point using pinhole model 
        X_3d_relative = (self.UMR_1_center_x - self.cx_cam1)*self.Z1[-1]/self.fx_cam1
        Y_3d_relative = ((self.UMR_1_center_y - self.cy_cam1)*self.Z1[-1]/self.fy_cam1)
        Z_3d_relative = ((self.UMR_2_center_y - self.cy_cam2)*self.Z2[-1]/self.fy_cam2)

        # The 3d trajectory data shifted to the origin point and in mm
        X_3d_next = (X_3d_relative-self.X_shift)
        Y_3d_next = -(Y_3d_relative-self.Y_shift)
        Z_3d_next = -(Z_3d_relative-self.Z_shift)

        # Append the new 3D coordinates to the lists
        self.X_3d.append(X_3d_next)
        self.Y_3d.append(Y_3d_next)
        self.Z_3d.append(Z_3d_next)

        # Adjust future depth estimates based on movement in Y and Z,
        # This is assuming smooth motion and small displacements
        Z1_next = self.cam1_to_box_distance+Z_3d_next
        Z2_next = self.cam2_to_box_distance+Y_3d_next
        self.Z1.append(Z1_next)
        self.Z2.append(Z2_next)
        #print(f"Z1={Z1_next:.4f} m, Z2={Z2_next:.4f} m")       

    def select_roi(self,cap):
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

    def update_roi_center(self,frame, roi):

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
                angle = ellipse[2]-90  # in degrees and make horizontal zero

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

    def angle_filter(self, angle_1_measured, angle_2_measured):
        #exponential moving average filter for angles
        if self.angle_1_filtered is None or self.angle_2_filtered is None:
            self.angle_1_filtered = angle_1_measured
            self.angle_2_filtered = angle_2_measured
        else:

            self.angle_1_filtered = self.alpha * angle_1_measured + (1 - self.alpha) * self.angle_1_filtered
            self.angle_2_filtered = self.alpha * angle_2_measured + (1 - self.alpha) * self.angle_2_filtered

        return self.angle_1_filtered, self.angle_2_filtered

    def find_nearest_trajectory_point(self, current_x, current_y):
        trajectory_xy = self.trajectory[:, :2]
        current_pos = np.array([current_x, current_y])
        distances = np.linalg.norm(trajectory_xy - current_pos, axis=1)
        index = np.argmin(distances)
        return index

    def controller(self):
        if self.trajectory_3d is None or len(self.trajectory_3d) == 0:
            return

        # Update the position of the kuka
        current_pos = np.array([self.X_3d[-1],self.Y_3d[-1],self.Z_3d[-1]])
        start_pos = np.array([self.X_3d[0],self.Y_3d[0],self.Z_3d[0]])

        delta_pos = current_pos-start_pos
        kuka_new_pos = self.kuka_start_pos + delta_pos

        # Update the orientation of the kuka
        current_angle = np.radians(self.angle_1)
        delta_rot = np.array([current_angle,0,0])
        kuka_new_rot = self.kuka_start_rot+delta_rot

        #send the new position and rotation
        self.send_pose(kuka_new_pos,kuka_new_rot)

        # # Find nearest trajectory point
        # idx = self.find_nearest_trajectory_point(current_x, current_y)
        # idx = min(idx, len(self.trajectory) - 1)

        # target_x, target_y, target_angle = self.trajectory[idx]

        # # 3. Compute control errors
        # error_pos = np.array([target_x - current_x, target_y - current_y])
        # #TODO here calculate needed change in angle with some law

        # # compute control signals (simple P-controller here, tune gains)
        # Kp_pos
        # Kp_angle

        # control_translation = Kp_pos * error_pos
        # control_rotation = Kp_angle * error_angle

        # self.pose = (current_x+self.initial_x_kuka,current_y+self.initial_y_kuka,self.initial_z_kuka,self.initial_alpha_kuka, self.initial_beta_kuka, steering_angle+self.initial_gamma_kuka)
        # self.send_pose()
        return 

    def calibrate_kuka(self):
        self.kuka_start_pos,self.kuka_start_rot = self.get_current_pose()
        self.calibrate_kuka_button.config(text="Kuka Calibrated",style='CalibrateKukaDone.TButton')

    def set_new_velocity(self):
        raw = self.velocity_entry.get()
        try:
            hz_velocity = float(raw)
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                f"“{raw}” is not a valid number.\nPlease enter a numeric value (e.g. 1.5)."
            )
            return

        rad_per_sec_velocity = hz_velocity * 2 * math.pi  # Convert Hz to rad/s
        self.motor_velocity_rad = rad_per_sec_velocity
        print(f"motor velocity is set to: {hz_velocity} Hz")

    def compute_initial_world_xyz(self):
        u = self.umr_1_center_x
        v = self.umr_1_center_y
        Z1 = self.Z1_initial  # camera1 → object distance (m)

        X_rel = (u - self.cx_cam1) * Z1 / self.fx_cam1
        Y_rel = (v - self.cy_cam1) * Z1 / self.fy_cam1

        # Shift into “box‐corner” world frame in X and Y
        X0 = X_rel - self.X_shift
        Y0 = -(Y_rel - self.Y_shift)

        # Compute Z relative to the box plane (i.e. how far above the box)
        Z0 = self.Z1_initial - self.cam1_to_box_distance

        return X0, Y0, Z0

    def generate_relative_linear_trajectory(self,length_mm=100,num_points=50,direction_deg=0.0):

        #get initial world‐XYZ (in meters), then convert to millimeters
        X0_m, Y0_m, Z0_m = self.compute_initial_world_xyz()
        X0 = X0_m * 1000.0
        Y0 = Y0_m * 1000.0
        Z0 = Z0_m * 1000.0

        #compute endpoint (in mm) along XY‐direction
        theta_rad = math.radians(direction_deg)
        dx = length_mm * math.cos(theta_rad)
        dy = length_mm * math.sin(theta_rad)
        X1 = X0 + dx
        Y1 = Y0 + dy

        # linearly interpolate X, Y; keep Z constant at Z0
        x_vals = np.linspace(X0, X1, num_points)
        y_vals = np.linspace(Y0, Y1, num_points)
        z_vals = np.full(num_points, Z0, dtype=np.float64)
        theta_array = np.full(num_points, direction_deg, dtype=np.float64)

        # Stack into (num_points × 4): [x_mm, y_mm, z_mm, θ_deg]
        trajectory_3d = np.vstack((x_vals, y_vals, z_vals, theta_array)).T

        return trajectory_3d

    def generate_curved_trajectory_3d(self,radius_mm=100,arc_angle_deg=90.0,num_points=50,direction_deg=0.0,turn_left=True):

        # Get initial world XYZ (in meters), convert to mm
        X0_m, Y0_m, Z0_m = self.compute_initial_world_xyz()
        X0 = X0_m * 1000.0
        Y0 = Y0_m * 1000.0
        Z0 = Z0_m * 1000.0  # height stays fixed

        #Convert initial heading to radians
        psi0 = math.radians(direction_deg)

        sweep_rad = math.radians(arc_angle_deg)
        if not turn_left:
            sweep_rad = -sweep_rad

        if turn_left:
            Xc = X0 - radius_mm * math.sin(psi0)
            Yc = Y0 + radius_mm * math.cos(psi0)
        else:
            Xc = X0 + radius_mm * math.sin(psi0)
            Yc = Y0 - radius_mm * math.cos(psi0)

        phi = np.linspace(0, sweep_rad, num_points)

        Xs = Xc + radius_mm * np.sin(psi0 + phi)
        Ys = Yc - radius_mm * np.cos(psi0 + phi)

        #Tangent heading at each waypoint = psi0 + phi + (±90°)
        if turn_left:
            theta_rad = psi0 + phi + (math.pi / 2)
        else:
            theta_rad = psi0 + phi - (math.pi / 2)

        # Convert to degrees, wrap into [0,360)
        theta_deg = (np.degrees(theta_rad)) % 360

        # Z stays fixed at Z0 for all points
        Zs = np.full(num_points, Z0, dtype=np.float64)
        trajectory_3d = np.vstack((Xs, Ys, Zs, theta_deg)).T
        return trajectory_3d
    
    def update_frame(self):
        
        # Check if the window is still open before updating
        if not self.window.winfo_exists():
            return
        
        #Read the frames
        # Read 30 frames ahead (skip 30 frames)
        for _ in range(30):
            ret1, frame1 = self.cap1.read()  # Read but discard 30 frames from camera 1
            ret2, frame2 = self.cap2.read()  # Read but discard 30 frames from camera 2

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
                self.UMR_1_bounding_box, self.UMR_1_roi,self.UMR_1_angle_measured, self.UMR_1_center_x, self.UMR_1_center_y = self.update_roi_center(frame1, self.UMR_1_roi)
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
                self.UMR_2_bounding_box,self.UMR_2_roi,self.UMR_2_angle_measured, self.UMR_2_center_x, self.UMR_2_center_y = self.update_roi_center(frame2, self.UMR_2_roi)
                x, y, w, h = [int(v) for v in self.UMR_2_roi]
                #cv2.rectangle(frame2_display, (x,y), (x+w,y+h), (0,0,255), 4) #This can draw the ROI around the umr, but it is not needed
                # Draw the bounding box around the umr
                cv2.drawContours(frame2_display, [self.UMR_2_bounding_box + (x, y)], 0, (0, 0, 255), 2)    
            frame2_resized = cv2.resize(frame2_display, (576, 324), interpolation=cv2.INTER_LINEAR)    
            frame_rgb2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
            img2 = ImageTk.PhotoImage(Image.fromarray(frame_rgb2))
            self.video_label2.imgtk = img2
            self.video_label2.config(image=img2)

        # Update the reconstructed pose and angle based on the UMRs and box positions
        if self.reconstruction_boolean is True:
            self.update_trajectory()
            if len(self.X_3d) % 5 == 0:  # only redraw every 5th frame
                self.update_trajectory_plot()

            self.angle_1, self.angle_2 = self.angle_filter(self.UMR_1_angle_measured, self.UMR_2_angle_measured)
            self.angle1_label.config(text=f"Angle 1: {self.angle_1:.2f}°")
            self.angle2_label.config(text=f"Angle 2: {self.angle_2:.2f}°")

        # Update the controller
        if self.controller_boolean is True:
            if self.reconstruction_boolean is False:
                print("First turn the reconstructor on")
                return
            self.controller()

        self.window.after(10, self.update_frame)

    def on_closing(self):
        if self.recording:
            self.out1.release()
            self.out2.release()
        self.cap1.release()
        self.cap2.release()
        self.window.destroy()
        self.kuka_python.shutdown_publisher()

    #def delta_change(self):
        #if change >1mm
        #     send.pose()
        # --> eerste boven hangen  en op vaste hoek houden

# Used if the recorder class is called seperately
if __name__ == "__main__":
    root = tk.Tk()
    app = ClosedLoopRecorder(root)
    root.mainloop()



    #TODO fix the angle calculation for the UMR's
    #TODO: klaarmaken voor live video door verwijderen: select_roi:verplaats naar eerste frame 
    #TODO: finish the closed loop controller and update the self.pose there 
    #TODO: door rechtdoor gaan de feedback tunen
    #TODO: feedforward bepalen door bachten maken 
    #TODO: en dan combinerne
    #TODO: kijken of ik pitch kan controleren ook