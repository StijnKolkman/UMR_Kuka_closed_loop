import time
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np 
import os
import math
import pandas as pd
import recorder_functions
import datetime
import json

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt

from kuka_python_node.kuka_node import start_kuka_node

"""
Dual-camera UMR tracker and KUKA closed-loop controller.

This application:
  • Captures video from two calibrated cameras and tracks an untethered magnetic robot (UMR).
  • Reconstructs the 3D trajectory of the UMR in real time.
  • Runs a closed-loop controller (pitch via forward/backward motion, yaw via rotation) 
    to command a KUKA robot that positions a rotating permanent magnet.
  • Provides a Tkinter GUI for calibration, recording, live visualization, and debugging.
  • Records synchronized video, 3D pose, and controller states for offline analysis.

Main dependencies: OpenCV, NumPy, Tkinter, Matplotlib, ROS (for KUKA communication).
"""

class ClosedLoopRecorder:
    def __init__(self, window):
        self.kuka_python = start_kuka_node()

        # Box to Kuka rotation matrix 
        self.R_BoxToKuka = np.array([
                [ 0, -1,  0],
                [-1,  0,  0],
                [ 0,  0, -1]
            ])

        # New send positions
        self.kuka_new_pos = None
        self.kuka_new_rot = None

        # reference trajectory
        self.trajectory_3d = None

        # Filtered angles
        self.angle_1_filtered = None
        self.angle_2_filtered = None
        self.alpha = 0.05
        self.i = 0

        # Controller state
        self.controller_boolean = False
        self.motor_velocity_rad = 0.0
        self.motor_boolean = False
        self.offset_angle = 0.0
        self.interval_time = 0.0

        #pitch settings
        self.last_time_controller = None
        self.pitch_setpoint_gain = 49.09 # Simple linear map. 16mm error will cause a 45 degrees pitch 
        self.integrator_pitch = 0.0
        self.integrator_pitch_max = 0.5  # Maximum integrator value to prevent windup
        self.integrator_pitch_min = -0.5  # Minimum integrator value to prevent windup
        self.Kp_pitch = 0.0256  # Proportional gain for pitch control --> this makes 45 degrees error into 0.02m moving forward
        self.Ki_pitch = 0.005  # Integral gain for pitch control. i just made it small
        self.pitch_compensation_min = -0.008
        self.pitch_compensation_max = +0.008

        #yaw settings
        self.look_ahead_offset = 0
        self.yaw_setpoint_gain = 52.00 # Simple linear map. 10mm error will cause a 30 degrees yaw 
        self.integrator_yaw = 0.0
        self.integrator_yaw_max = 0.5  # Maximum integrator value to prevent windup
        self.integrator_yaw_min = -0.5  # Minimum integrator value to prevent windup
        self.Kp_yaw = 1  # Proportional gain for yaw control --> this makes 30 degrees error into 30degrees input moving forward
        self.Ki_yaw = 0.1  # 
        self.yaw_compensation_min = np.radians(-30) # Minimum yaw compensation in radians
        self.yaw_compensation_max = np.radians(30)  # Maximum yaw compensation in radians

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
        self.X_3d = None
        self.Y_3d = None
        self.Z_3d = None
        self.N_frames = 0

        # Known box dimensions (mm)
        self.real_box_width_cam1_mm = 108
        self.real_box_height_cam1_mm = 56
        self.real_box_width_cam2_mm = 108
        self.real_box_height_cam2_mm = 32
        
        # recording states 
        self.recording = False
        self.recorded_file_names = None
        self.frames_cam1 = []
        self.frames_cam2 = []
        self.record_start_time = None
        self.timestamps = []
        self.X_3d_recording = []
        self.Y_3d_recording = []
        self.Z_3d_recording = []

        # Controller recording parameters
        self.pitch_setpoint_rec   = []
        self.current_pitch_rec    = []
        self.pitch_error_rec      = []
        self.pitch_comp_rec       = []
        self.pitch_comp_iterm_rec = []
        self.pitch_comp_pterm_rec = []

        self.current_yaw_rec      = []
        self.yaw_setpoint_rec     = []
        self.yaw_error_rec        = []
        self.yaw_pterm_rec        = []
        self.yaw_iterm_rec        = []
        self.yaw_comp_rec         = []
        


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

        # ALL THE GUI RELATED SETTINGS
        self.window = window
        self.window.title("Dual Camera Recorder")
        self.window.geometry("1024x700")
        self.window.minsize(800, 600)
        self.window.grid_columnconfigure((0,1,2), weight=1)
        self.window.grid_rowconfigure(0, weight=1)

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
        style.configure('MotorOff.TButton', background='red', foreground='white')
        style.configure('MotorOn.TButton',  background='green', foreground='white')

        # ALL THE VIDEO RELATED SETTINGS
        #self.cap1 = cv2.VideoCapture(r"/home/ram-micro/Documents/Stijn/UMR_Kuka_closed_loop/test_50deg_02hz/test_50deg_02hz_cam1.mp4")
        #self.cap2 = cv2.VideoCapture(r"/home/ram-micro/Documents/Stijn/UMR_Kuka_closed_loop/test_50deg_02hz/test_50deg_02hz_cam2.mp4")
        self.cap1 = cv2.VideoCapture(6, cv2.CAP_V4L2) 
        self.cap2 = cv2.VideoCapture(4, cv2.CAP_V4L2)
        self.cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

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

        self.video_label1 = ttk.Label(self.cam1_frame)
        self.video_label1.pack(fill="both", expand=True)
        self.video_label2 = ttk.Label(self.cam2_frame)
        self.video_label2.pack(fill="both", expand=True)

        # Matplotlib 3D plot + toolbar
        self.fig = plt.Figure(figsize=(6, 5))
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()

        # Build sub-panels in controls_frame
        # Recording group
        self.rec_frame = ttk.Labelframe(self.controls_frame, text="Recording", labelanchor="n", padding=(10,8))
        self.rec_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.rec_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.rec_frame, text="File name:").grid(row=0, column=0, sticky="w", padx=5, pady=(0,4))
        self.filename_entry = ttk.Entry(self.rec_frame)
        self.filename_entry.insert(0, "Recording")
        self.filename_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=(0,4))
        self.record_button = ttk.Button(self.rec_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4,0))

        # Calibration group
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
        self.recorded_files_label.grid(row=5, column=0, sticky="w", pady=(4,0))

        self.reset_angle = ttk.Button(self.cal_frame, text="Set angles to initial position", command=self.reset_all_angles)
        self.reset_angle.grid(row=4, column=0, sticky="ew", pady=2)

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

        # Motor toggle button
        self.motor_button = ttk.Button(self.ctrl_frame,text="Motor: OFF",command=self.toggle_motor,style='MotorOff.TButton')
        self.motor_button.grid(row=2, column=0, columnspan=3, sticky="ew", pady=2)

        # Angle displays
        self.angle1_label = ttk.Label(self.ctrl_frame, text="Angle 1: ---°")
        self.angle1_label.grid(row=3, column=0, sticky="w", padx=5, pady=(6,0))
        self.angle2_label = ttk.Label(self.ctrl_frame, text="Angle 2: ---°")
        self.angle2_label.grid(row=3, column=1, sticky="w", padx=5, pady=(6,0))

        # Cam 1 focus
        self.focus_label1 = ttk.Label(self.ctrl_frame, text="Focus Cam 1:")
        self.focus_label1.grid(row=4, column=0, sticky="w", padx=5, pady=(6,0))
        self.focus_slider1 = ttk.Scale(self.ctrl_frame, from_=0, to=255, orient='horizontal', command=lambda val: self.set_focus(self.cap1, "focus_value_label1", val))
        self.focus_slider1.set(91)
        self.focus_slider1.grid(row=4, column=1, columnspan=2, sticky="ew", padx=5, pady=(6,0))

        # Cam 2 focus
        self.focus_label2 = ttk.Label(self.ctrl_frame, text="Focus Cam 2:")
        self.focus_label2.grid(row=5, column=0, sticky="w", padx=5, pady=(4,0))
        self.focus_slider2 = ttk.Scale(self.ctrl_frame, from_=0, to=255, orient='horizontal', command=lambda val: self.set_focus(self.cap2, "focus_value_label2", val))
        self.focus_slider2.set(58)
        self.focus_slider2.grid(row=5, column=1, columnspan=2, sticky="ew", padx=5, pady=(4,0))

        # Yaw info frame (to the left of Pitch)
        self.yaw_info = ttk.Labelframe(self.ctrl_frame, text="Yaw Debug", padding=(6,4))
        self.yaw_info.grid(row=6, column=0, sticky="ew", pady=(8,0), padx=(0,5))
        self.yaw_info.grid_columnconfigure(1, weight=1)

        # Row 0: current yaw
        ttk.Label(self.yaw_info, text="Current Yaw:").grid(row=0, column=0, sticky="w")
        self.current_yaw_label = ttk.Label(self.yaw_info, text="--- rad")
        self.current_yaw_label.grid(row=0, column=1, sticky="w")

        # Row 1: yaw setpoint (if you have one)
        ttk.Label(self.yaw_info, text="Yaw Setpoint:").grid(row=1, column=0, sticky="w")
        self.yaw_setpoint_label = ttk.Label(self.yaw_info, text="--- rad")
        self.yaw_setpoint_label.grid(row=1, column=1, sticky="w")

        # Row 2: yaw error
        ttk.Label(self.yaw_info, text="Yaw Error:").grid(row=2, column=0, sticky="w")
        self.yaw_error_label = ttk.Label(self.yaw_info, text="--- rad")
        self.yaw_error_label.grid(row=2, column=1, sticky="w")

        # Row 3: yaw P-term (if you add one later)
        ttk.Label(self.yaw_info, text="P-term:").grid(row=3, column=0, sticky="w")
        self.yaw_pterm_label = ttk.Label(self.yaw_info, text="---")
        self.yaw_pterm_label.grid(row=3, column=1, sticky="w")

        # Row 4: yaw I-term (optional)
        ttk.Label(self.yaw_info, text="I-term:").grid(row=4, column=0, sticky="w")
        self.yaw_iterm_label = ttk.Label(self.yaw_info, text="---")
        self.yaw_iterm_label.grid(row=4, column=1, sticky="w")

        # Row 5: total yaw compensation or delta_rot[1]
        ttk.Label(self.yaw_info, text="Compensation:").grid(row=5, column=0, sticky="w")
        self.yaw_comp_label = ttk.Label(self.yaw_info, text="---")
        self.yaw_comp_label.grid(row=5, column=1, sticky="w")

        # Pitch info frame
        self.pitch_info = ttk.Labelframe(self.ctrl_frame, text="Pitch Debug", padding=(6,4))
        self.pitch_info.grid(row=6, column=1, sticky="ew", pady=(8,0))
        self.pitch_info.grid_columnconfigure(1, weight=1)

        # Row 0: current pitch
        ttk.Label(self.pitch_info, text="Current Pitch:").grid(row=0, column=0, sticky="w")
        self.current_pitch_label = ttk.Label(self.pitch_info, text="--- rad")
        self.current_pitch_label.grid(row=0, column=1, sticky="w")

        # Row 1: pitch setpoint
        ttk.Label(self.pitch_info, text="Pitch Setpoint:").grid(row=1, column=0, sticky="w")
        self.pitch_setpoint_label = ttk.Label(self.pitch_info, text="--- rad")
        self.pitch_setpoint_label.grid(row=1, column=1, sticky="w")

        # Row 2: error
        ttk.Label(self.pitch_info, text="Error:").grid(row=2, column=0, sticky="w")
        self.pitch_error_label = ttk.Label(self.pitch_info, text="--- rad")
        self.pitch_error_label.grid(row=2, column=1, sticky="w")

        # Row 3: P‐term
        ttk.Label(self.pitch_info, text="P-term:").grid(row=3, column=0, sticky="w")
        self.pitch_pterm_label = ttk.Label(self.pitch_info, text="---")
        self.pitch_pterm_label.grid(row=3, column=1, sticky="w")

        # Row 4: I-term
        ttk.Label(self.pitch_info, text="I-term:").grid(row=4, column=0, sticky="w")
        self.pitch_iterm_label = ttk.Label(self.pitch_info, text="---")
        self.pitch_iterm_label.grid(row=4, column=1, sticky="w")

        # Row 5: total compensation
        ttk.Label(self.pitch_info, text="Compensation:").grid(row=5, column=0, sticky="w")
        self.pitch_comp_label = ttk.Label(self.pitch_info, text="---")
        self.pitch_comp_label.grid(row=5, column=1, sticky="w")

        # Start the update loop and handle window‐close
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def set_focus(self, cap, label_attr: str, val):
        """Set focus on a given camera and update its label if available."""

        focus_value = float(val)
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)

        if hasattr(self, label_attr):
            getattr(self, label_attr).config(text=f"Focus Value: {focus_value:.2f}")

    def toggle_controller(self):
        """Toggle controller ON/OFF and update button + motor state."""

        # flip state
        self.controller_boolean = not self.controller_boolean

        # require trajectory + reconstructor active
        if self.trajectory_3d is None or not self.reconstruction_boolean:
            messagebox.showerror(
                "Error",
                "First start the 3D reconstructor and make sure a reference trajectory is made"
            )
            # reset toggle back to OFF
            self.controller_boolean = False
            self.controller_button.config(text="Controller: OFF", style="ControllerOff.TButton")
            self.kuka_python.set_motor_speed(0.0)
            return

        if self.controller_boolean:
            self.controller_button.config(text="Controller: ON", style="ControllerOn.TButton")
            self.kuka_python.set_motor_speed(self.motor_velocity_rad)
        else:
            self.controller_button.config(text="Controller: OFF", style="ControllerOff.TButton")
            self.kuka_python.set_motor_speed(0.0)

    def toggle_motor(self):
        """Turn the KUKA motor on or off, independently of the controller logic."""

        # flip state
        self.motor_boolean = not self.motor_boolean

        try:
            if self.motor_boolean:
                self.motor_button.config(text="Motor: ON", style="MotorOn.TButton")
                self.kuka_python.set_motor_speed(self.motor_velocity_rad)
            else:
                self.motor_button.config(text="Motor: OFF", style="MotorOff.TButton")
                self.kuka_python.set_motor_speed(0.0)
        except Exception as e:
            messagebox.showerror("Error", f"Motor command failed: {e}")

            # reset to safe OFF state
            self.motor_boolean = False
            self.motor_button.config(text="Motor: OFF", style="MotorOff.TButton")
            self.kuka_python.set_motor_speed(0.0)

    def toggle_recording(self):
        """Start/stop dual‑camera recording and save data/params when stopping."""

        if self.reconstruction_boolean is False:
            messagebox.showerror("Error","First start the reconstruction to record the trajectory")
            return 

        self.recording = not self.recording

        filename = self.filename_entry.get().strip() or "recording"
        output_dir = os.path.join(os.getcwd(), filename)
        os.makedirs(output_dir, exist_ok=True)

        cam1_filename = os.path.join(output_dir, f"{filename}_cam1.mp4")
        cam2_filename = os.path.join(output_dir, f"{filename}_cam2.mp4")

        if self.recording:
            # Start recording
            self.N_frames = 0
            self.record_start_time = time.time()
            self.recorded_files_label.config(text="Recording in progress...")
            #print(f"Started recording: {cam1_filename} & {cam2_filename}")
            self.record_button.configure(text="Stop recording", style='RecordingOn.TButton')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out1 = cv2.VideoWriter(cam1_filename, fourcc, 20,(1280, 720))
            self.out2 = cv2.VideoWriter(cam2_filename, fourcc, 20, (1280, 720))


            if not self.out1.isOpened() or not self.out2.isOpened():
                print(f"Error: VideoWriter for {cam1_filename} or {cam2_filename} failed to open")
                return

        else:
            # Stop recording
            duration = time.time() - self.record_start_time
            fps_value = self.N_frames / duration if duration > 0 else 30.0
            print(f"Duration: {duration:.2f}s — FPS: {fps_value:.2f}")

            self.out1.release()
            self.out2.release()

            files_text = f"Recorded files:\n{cam1_filename}\n{cam2_filename}"
            self.recorded_files_label.config(text=files_text)
            print("Recording done and saved")
            self.recorded_file_names = (cam1_filename, cam2_filename)  # Store filenames
            self.record_button.configure(text="Start recording", style='RecordingOff.TButton')
            output_file_path = os.path.join(output_dir, f"{filename}_trajectory.csv")

            if self.X_3d is None or not self.timestamps:
                messagebox.showerror("Error",
                    "No trajectory to save, only saved the videos.")
                return
            cols = {
                'Time'              : self.timestamps,
                'X'                 : self.X_3d_recording,
                'Y'                 : self.Y_3d_recording,
                'Z'                 : self.Z_3d_recording,

                # ─ pitch channels ─
                'pitch_setpoint'    : self.pitch_setpoint_rec,
                'current_pitch'     : self.current_pitch_rec,
                'pitch_error'       : self.pitch_error_rec,
                'pitch_comp'        : self.pitch_comp_rec,
                'pitch_comp_iterm'  : self.pitch_comp_iterm_rec,
                'pitch_comp_pterm'  : self.pitch_comp_pterm_rec,

                # ─ yaw channels ─
                'current_yaw'       : self.current_yaw_rec,
                'yaw_setpoint'      : self.yaw_setpoint_rec,
                'yaw_error'         : self.yaw_error_rec,
                'yaw_pterm'         : self.yaw_pterm_rec,
                'yaw_iterm'         : self.yaw_iterm_rec,
                'yaw_comp'          : self.yaw_comp_rec,
            }

            # Ensure all lists are of equal length
            target = len(self.timestamps)
            for lst in cols.values():
                if len(lst) < target:
                    lst.extend([float('nan')] * (target - len(lst)))
                elif len(lst) > target:
                    del lst[target:]
                    
            pd.DataFrame(cols).to_csv(output_file_path, index=False)
            print(f"[INFO] Full trajectory + controller channels saved to {output_file_path}")

            # Save the controller parameters to a JSON file
            param_dump = {
                "timestamp_iso"        : datetime.datetime.now().isoformat(timespec="seconds"),
                "motor_velocity_rad"   : self.motor_velocity_rad,
                "pitch_setpoint_gain"  : self.pitch_setpoint_gain,
                "Kp_pitch"             : self.Kp_pitch,
                "Ki_pitch"             : self.Ki_pitch,
                "integrator_limits"    : [self.integrator_pitch_min, self.integrator_pitch_max],
                "alpha_angle_filter"   : self.alpha,
                "R_BoxToKuka"          : self.R_BoxToKuka.tolist(),
                "Box-size"             : [self.real_box_width_cam1_mm,self.real_box_height_cam1_mm,self.real_box_width_cam2_mm,self.real_box_height_cam2_mm]
            }

            json_path = os.path.join(output_dir, f"{filename}_controller_params.json")
            with open(json_path, "w") as f:
                json.dump(param_dump, f, indent=4)
            print(f"[INFO] Controller parameters saved to {json_path}")

            # Initiate timestamps array for next recording
            self.timestamps = []

    def toggle_calibration_cam1(self):
        """Select ROIs for box and UMR in camera 1."""
        self.box_1_roi = recorder_functions.select_roi(self.cap1)
        self.UMR_1_roi = recorder_functions.select_roi(self.cap1)
        self.calibration_button1.config(text="Cam 1 Calibrated", style="Calibrated.TButton")

    def toggle_calibration_cam2(self):
        """Select ROIs for box and UMR in camera 2."""
        self.box_2_roi = recorder_functions.select_roi(self.cap2)
        self.UMR_2_roi = recorder_functions.select_roi(self.cap2)
        self.calibration_button2.config(text="Cam 2 Calibrated", style="Calibrated.TButton")

    def reset_all_angles(self):
        """Reset KUKA rotation to its calibrated starting angle."""
        if self.kuka_start_rot is None or self.kuka_new_pos is None or self.kuka_new_pos is None:
            messagebox.showerror("Reset Error", "No starting angle defined, first calibrate the KUKA!")
            return 
        self.kuka_new_rot = self.kuka_start_rot
        self.send_pose(self.kuka_new_pos,self.kuka_new_rot)

    def start_trajectory_reconstruction(self):
        """Initialize 3D trajectory reconstruction using calibrated ROIs from both cameras."""

        if self.box_1_roi is None or self.box_2_roi is None or self.UMR_1_roi is None or self.UMR_2_roi is None:
            messagebox.showerror("Error", "Camera('s) not calibrated, first calibrate camera 1 and 2!")
            return


        self.reconstruction_boolean = True
        
        # extract intrinsics
        self.fx_cam1, self.fy_cam1 = self.camera_matrix1[0, 0], self.camera_matrix1[1, 1]
        self.fx_cam2, self.fy_cam2 = self.camera_matrix2[0, 0], self.camera_matrix2[1, 1]
        self.cx_cam1, self.cy_cam1 = self.camera_matrix1[0, 2], self.camera_matrix1[1, 2]
        self.cy_cam2 = self.camera_matrix2[1, 2]

        # reset trajectory arrays
        self.Z1, self.Z2, self.X_3d, self.Y_3d, self.Z_3d = [], [], [], [], []

        # unpack box ROIs
        self.box_1_x, self.box_1_y, self.box_1_width_px, self.box_1_height_px = self.box_1_roi
        self.box_2_x, self.box_2_y, self.box_2_width_px, self.box_2_height_px = self.box_2_roi

        # compute UMR centers
        x, y, w, h = self.UMR_1_roi
        self.umr_1_center_x, self.umr_1_center_y = x + w // 2, y + h // 2
        x, y, w, h = self.UMR_2_roi
        self.umr_2_center_x, self.umr_2_center_y = x + w // 2, y + h // 2

        # pixel-to-mm scaling
        self.mm_per_pixel_cam_1_x = self.real_box_width_cam1_mm / self.box_1_width_px 
        self.mm_per_pixel_cam_1_y = self.real_box_height_cam1_mm / self.box_1_height_px
        self.mm_per_pixel_cam_2_x = self.real_box_width_cam2_mm / self.box_2_width_px 
        self.mm_per_pixel_cam_2_y = self.real_box_height_cam2_mm / self.box_2_height_px

        # distances from cameras to box
        self.cam1_to_box_distance = recorder_functions.camera_to_box_distance(self.real_box_width_cam1_mm,self.box_1_width_px,self.fx_cam1)
        self.cam2_to_box_distance = recorder_functions.camera_to_box_distance(self.real_box_width_cam2_mm,self.box_2_width_px,self.fx_cam2)

        # initial offsets
        bottom_box_cam1 = self.box_1_y+self.box_1_height_px
        bottom_box_cam2 = self.box_2_y+self.box_2_height_px
        self.initial_y = (bottom_box_cam1-self.umr_1_center_y)*(self.mm_per_pixel_cam_1_y/1000)
        self.initial_z = (bottom_box_cam2-self.umr_2_center_y)*(self.mm_per_pixel_cam_2_y/1000)

        # Calculate the initial Z position based on the box height and camera distance
        self.Z1_initial = self.cam1_to_box_distance + self.initial_z
        self.Z2_initial = self.cam2_to_box_distance + self.initial_y
        self.Z1.append(self.Z1_initial)
        self.Z2.append(self.Z2_initial)

        # origin shifts (box corner → world origin)
        # Now we assume that with the camera setup, the left bottom corner of camera 1 is the origin. In camera 2 this is the bottom right corner
        x_box_1_px, y_box_1_px = self.box_1_x, self.box_1_y
        origin_box1_x_px = x_box_1_px
        origin_box1_y_px = y_box_1_px + self.box_1_height_px
        Z_box = self.cam1_to_box_distance
        self.X_shift = (origin_box1_x_px - self.cx_cam1) * Z_box / self.fx_cam1
        self.Y_shift = (origin_box1_y_px - self.cy_cam1) * Z_box / self.fy_cam1

        x_box_2_px, y_box_2_px = self.box_2_x, self.box_2_y
        origin_box2_y_px = y_box_2_px + self.box_2_height_px
        Z_box = self.cam2_to_box_distance
        self.Z_shift = (origin_box2_y_px - self.cy_cam2) * Z_box / self.fy_cam2

        # now make the reference trajectory! linear or curved
        X0, Y0, Z0 = self.compute_initial_world_xyz()
        #self.trajectory_3d = recorder_functions.generate_relative_linear_trajectory_3d(X0,Y0,Z0,length_m=0.1,num_points=50,direction_rad=0.0)

        # alternatively:
        #self.trajectory_3d = recorder_functions.generate_curved_trajectory_3d(X0,Y0,Z0,radius_m=0.1,arc_angle_rad=math.pi/2,num_points=50,direction_rad=0.0,turn_left=True)
        self.trajectory_3d = recorder_functions.generate_sine_trajectory_3d(
        X0, Y0, Z0,
        length_x=1.0,        # total distance to travel in X
        amplitude=0.01,       # peak Y deviation (sine amplitude)
        wavelength=0.1,      # wavelength of the sine (in meters of X)
        num_points=200
    )
        recorder_functions.save_trajectory_to_csv(self.trajectory_3d,self.filename_entry)

    def update_trajectory_plot(self):
        """Update the 3D trajectory plot with measured path, planned path, and calibration box."""

        self.ax3d.clear()

        # 3D trajectory data to mm
        X = (np.array(self.X_3d))*1000
        Y = (np.array(self.Y_3d))*1000
        Z = (np.array(self.Z_3d))*1000 

        #Print trajectory points
        #print(f"X={X[-1]:.2f} mm, Y={Y[-1]:.2f} mm, Z={Z[-1]:.2f} mm")

        # Plot measdured trajectory if available
        if X.size > 0:
            self.ax3d.plot(X, Y, Z, color='blue', linewidth=4, label='Trajectory')
        
        # plot planned trajectory if available
        if hasattr(self, 'trajectory_3d'):
            planned = np.array(self.trajectory_3d)
            planned_X = planned[:, 0]*1000 
            planned_Y = planned[:, 1]*1000 
            planned_Z = planned[:, 2]*1000 
            self.ax3d.plot(planned_X, planned_Y, planned_Z, label="Planned",color='black',linestyle='--')

        # calibration box dimensions (mm)
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

        # plot calibration box
        for start, end in edges:
            xs = [corners[start,0], corners[end,0]]
            ys = [corners[start,1], corners[end,1]]
            zs = [corners[start,2], corners[end,2]]
            self.ax3d.plot(xs, ys, zs, color='red', linestyle='--', linewidth=1, label='Box' if start==0 and end==1 else "")

        # axis labels and title
        self.ax3d.set_title("3D Trajectory with Calibration Box")
        self.ax3d.set_xlabel("X (mm)")
        self.ax3d.set_ylabel("Y (mm)")
        self.ax3d.set_zlabel("Z (mm)")

        # enforce equal aspect ratio
        max_range = np.array([w, h, d]).max() / 2.0
        mid_x = w / 2.0
        mid_y = h / 2.0
        mid_z = d / 2.0

        self.ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax3d.set_ylim(mid_y + max_range, mid_y - max_range) #flip Y --> i am stupid and used left hand coordinate system when defining the camera :(
        self.ax3d.set_zlim(mid_z - max_range, mid_z + max_range)


        #self.ax3d.view_init(elev=90, azim=270) #--> view cam 1
        #self.ax3d.view_init(elev=30, azim=250) 

        self.ax3d.legend()
        self.canvas.draw()

    def send_pose(self,pos,rot):
        """Publish a pose to ROS (position in m, rotation in rad)."""
        if pos is None or rot is None:
            print("No pose to publish.")
            return

        # convert to mm and degrees
        pos_mm = np.array(pos, dtype=float) * 1000
        rot_deg = np.degrees(rot)

        pose = np.concatenate((pos_mm, rot_deg))
        self.kuka_python.publish_pose(pose)
        # print(f"Published pose: {pose}")
 
    def get_current_pose(self):
        """"get the current pose of the kuka in m and radians"""
        kuka_pose = self.kuka_python.get_position()
        pos = np.array(kuka_pose[0:3], dtype=float)
        rot = np.array(kuka_pose[3:6], dtype=float)
        return pos, rot

    def update_trajectory(self):
        """Update 3D trajectory from the latest UMR centers (undistort → back-project → shift)."""
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

         # apply world-frame shifts/signs
        X_3d_next = (X_3d_relative-self.X_shift)
        Y_3d_next = (-Y_3d_relative+self.Y_shift)
        Z_3d_next = -(Z_3d_relative+self.Z_shift)

        # Append the new 3D coordinates to the lists
        self.X_3d.append(X_3d_next)
        self.Y_3d.append(Y_3d_next)
        self.Z_3d.append(Z_3d_next)

        # Adjust future depth estimates based on movement in Y and Z, This is assuming smooth motion and small displacements
        Z1_next = self.cam1_to_box_distance+Z_3d_next
        Z2_next = self.cam2_to_box_distance+Y_3d_next
        self.Z1.append(Z1_next)
        self.Z2.append(Z2_next)    

    def angle_filter(self, angle_1_measured, angle_2_measured):
        """Exponential moving average filter for two measured angles."""
        if self.angle_1_filtered is None or self.angle_2_filtered is None:
            self.angle_1_filtered = angle_1_measured
            self.angle_2_filtered = angle_2_measured
        else:

            self.angle_1_filtered = self.alpha * angle_1_measured + (1 - self.alpha) * self.angle_1_filtered
            self.angle_2_filtered = self.alpha * angle_2_measured + (1 - self.alpha) * self.angle_2_filtered
        return self.angle_1_filtered, self.angle_2_filtered

    def controller(self,time_controller):
        """Main closed-loop controller: computes pitch/yaw compensation and sends pose to KUKA."""
        if self.trajectory_3d is None or len(self.trajectory_3d) == 0:
            return
        
        # TIME UPDATE -----------------------------------------------------------------------------
        if self.last_time_controller is None:
            self.last_time_controller = time_controller
            return
        else:
            dt = time_controller - self.last_time_controller
            self.last_time_controller = time_controller
        
        # PITCH CONTROLLER-------------------------------------------------------------------------
        # pitch is controlled by the kuka moving in front or back of the umr

        # pick the nearest reference point in the trajectory
        index_nearest_reference_point = recorder_functions.find_nearest_trajectory_point(self.trajectory_3d,self.X_3d[-1],self.Y_3d[-1])
        z_nearest_ref = self.trajectory_3d[index_nearest_reference_point, 2]

        # calculate pitch setpoint based on the nearest reference point
        dz = self.Z_3d[-1] - z_nearest_ref
        pitch_setpoint = self.pitch_setpoint_gain * dz 
        pitch_setpoint = max(min(pitch_setpoint,  math.pi/4), -math.pi/4) # limit the pitch setpoint to ±45 degrees

        # Get the error in pitch
        current_pitch = -self.angle_2_filtered
        error_pitch = pitch_setpoint - current_pitch

        # Proportional and Integral control for pitch
        pitch_compensation_pterm = self.Kp_pitch * error_pitch 

        self.integrator_pitch += error_pitch * dt
        if self.integrator_pitch_min is not None:
            self.integrator_pitch = max(self.integrator_pitch_min, self.integrator_pitch)
        if self.integrator_pitch_max is not None:
            self.integrator_pitch = min(self.integrator_pitch_max, self.integrator_pitch)
        pitch_compensation_iterm = self.Ki_pitch * self.integrator_pitch 
        
        pitch_compensation = pitch_compensation_pterm + pitch_compensation_iterm
        pitch_compensation= max(self.pitch_compensation_min, min(pitch_compensation, self.pitch_compensation_max))

        # apply pitch compensation by moving the kuka in front or back of the UMR
        delta_pos_pitch_compensation = np.array([np.cos(abs(self.angle_1_filtered))*pitch_compensation, np.sin(self.angle_1_filtered)*-pitch_compensation, 0]) 

        # YAW Controller--------------------------------------------------------------------------
        # yaw is controlled by the kuka rotating around the Z axis

        # pick the nearest reference point in the trajectory plus and look-ahead offset and find the yaw angle at the reference point
        index_ahead = index_nearest_reference_point+self.look_ahead_offset
        if index_nearest_reference_point + self.look_ahead_offset >= len(self.trajectory_3d):
            index_ahead = len(self.trajectory_3d) - 1
        y_nearest_ref = self.trajectory_3d[index_ahead, 1]
        yaw_trajectory = self.trajectory_3d[index_ahead, 3]

        # calculate the yaw correction based on the distance to the traje_ctory
        dy = self.Y_3d[-1] - y_nearest_ref
        yaw_correction = self.yaw_setpoint_gain * dy
        yaw_setpoint = yaw_trajectory + yaw_correction  #The desired yaw angle is the trajectory yaw plus the correction
        yaw_setpoint = max(min(yaw_setpoint,  math.pi/2), -math.pi/2) # limit the yaw setpoint to ±90 degrees 

        # Get the error in yaw
        current_yaw = self.angle_1_filtered  # Assuming angle_1 is the yaw angle
        error_yaw = yaw_setpoint - current_yaw

        # Proportional and Integral control for yaw   
        yaw_compensation_pterm = self.Kp_yaw * error_yaw

        self.integrator_yaw += error_yaw * dt
        if self.integrator_yaw_min is not None:
            self.integrator_yaw = max(self.integrator_yaw_min, self.integrator_yaw)
        if self.integrator_yaw_max is not None:
            self.integrator_yaw = min(self.integrator_yaw_max, self.integrator_yaw)
        yaw_compensation_iterm = self.Ki_yaw * self.integrator_yaw

        yaw_compensation = yaw_compensation_pterm + yaw_compensation_iterm
        yaw_compensation= max(self.yaw_compensation_min, min(yaw_compensation, self.yaw_compensation_max))

        # update the rotation of the kuka
        delta_rot = np.array([0, self.angle_1_filtered+yaw_compensation, 0])  # Apply yaw compensation by rotating the kuka

        # KUKA POSITION UPDATE -----------------------------------------------------------------
        current_pos = np.array([self.X_3d[-1],self.Y_3d[-1],-self.Z_3d[-1]])
        start_pos = np.array([self.X_3d[0],self.Y_3d[0],-self.Z_3d[0]])
        delta_pos = current_pos-start_pos + delta_pos_pitch_compensation

        #rotate the rot and pos
        delta_pos, delta_rot = recorder_functions.transform_pose(self.R_BoxToKuka,delta_pos, delta_rot)

        #apply to calibrated start position
        self.kuka_new_pos = self.kuka_start_pos+delta_pos
        self.kuka_new_rot = self.kuka_start_rot+delta_rot

        #send the new position and rotation
        self.send_pose(self.kuka_new_pos,self.kuka_new_rot)


        # GUI UPDATE ----------------------------------------------------------------------------
        self.current_yaw_label .config(text=f"{current_yaw:.3f} rad")
        self.yaw_setpoint_label.config(text=f"{yaw_setpoint:.3f} rad")
        self.yaw_error_label  .config(text=f"{error_yaw:.3f} rad")
        self.yaw_pterm_label  .config(text=f"{yaw_trajectory :.3f}")
        self.yaw_iterm_label  .config(text=f"{yaw_correction :.3f}")
        self.yaw_comp_label   .config(text=f"{yaw_compensation:.3f}")

        self.current_pitch_label .config(text=f"{current_pitch:.3f} rad")
        self.pitch_setpoint_label.config(text=f"{pitch_setpoint:.3f} rad")
        self.pitch_error_label   .config(text=f"{error_pitch:.3f} rad")
        self.pitch_pterm_label   .config(text=f"{pitch_compensation_pterm:.3f}")
        self.pitch_iterm_label   .config(text=f"{pitch_compensation_iterm:.3f}")
        self.pitch_comp_label    .config(text=f"{pitch_compensation:.3f}")

        # RECORDING UPDATE------------------------------------------------------------------------
        if self.recording:
            # pitch channels
            self.pitch_setpoint_rec.append(pitch_setpoint)
            self.current_pitch_rec.append(current_pitch)
            self.pitch_error_rec.append(error_pitch)
            self.pitch_comp_rec.append(pitch_compensation)
            self.pitch_comp_iterm_rec.append(pitch_compensation_iterm)
            self.pitch_comp_pterm_rec.append(pitch_compensation_pterm)

            # yaw channels
            self.current_yaw_rec.append(current_yaw)  
            self.yaw_setpoint_rec.append(yaw_setpoint)
            self.yaw_error_rec.append(error_yaw)
            self.yaw_pterm_rec.append(yaw_compensation_pterm)
            self.yaw_iterm_rec.append(yaw_compensation_iterm)
            self.yaw_comp_rec.append(yaw_compensation) 
        return 

    def straight_controller(self,time_controller):
        ''' old controller, only pitch compensation, no yaw compensation'''
        if self.trajectory_3d is None or len(self.trajectory_3d) == 0:
            return
        
        # update dt --> needed for the controller
        if self.last_time_controller is None:
            self.last_time_controller = time_controller
            return
        else:
            dt = time_controller - self.last_time_controller
            self.last_time_controller = time_controller
        
        # PITCH CONTROLLER --> pitch is controlled by the kuka moving in front or back of the umr
        index_nearest_reference_point = recorder_functions.find_nearest_trajectory_point(self.trajectory_3d,self.X_3d[-1],self.Y_3d[-1])
        z_nearest_ref = self.trajectory_3d[index_nearest_reference_point, 2]

        dz = self.Z_3d[-1] - z_nearest_ref  # Calculate the difference in Z position 
        pitch_setpoint = self.pitch_setpoint_gain * dz # minus‐sign so negative error ⇒ positive (nose‐up)
        pitch_setpoint = max(min(pitch_setpoint,  math.pi/4), -math.pi/4)
        current_pitch = -self.angle_2_filtered  # Assuming angle_2 is the pitch angle

        error_pitch = pitch_setpoint - current_pitch
        pitch_compensation_pterm = self.Kp_pitch * error_pitch  # Proportional control for pitch

        self.integrator_pitch += error_pitch * dt  # Update the integrator for pitch
        if self.integrator_pitch_min is not None:
            self.integrator_pitch = max(self.integrator_pitch_min, self.integrator_pitch)
        if self.integrator_pitch_max is not None:
            self.integrator_pitch = min(self.integrator_pitch_max, self.integrator_pitch)
        pitch_compensation_iterm = self.Ki_pitch * self.integrator_pitch  # Integral control for pitch

        pitch_compensation = pitch_compensation_pterm + pitch_compensation_iterm
        pitch_compensation= max(self.pitch_compensation_min, min(pitch_compensation, self.pitch_compensation_max))
        delta_pos_pitch_compensation = np.array([pitch_compensation, 0, 0])  # Apply pitch compensation by moving the kuka in front or back of the UMR
        print(delta_pos_pitch_compensation)

        delta_rot = np.array([0, 0,0])

        # Update the position of the kuka
        current_pos = np.array([self.X_3d[-1],self.Y_3d[-1],-self.Z_3d[-1]])
        start_pos = np.array([self.X_3d[0],self.Y_3d[-1],-self.Z_3d[0]])
        delta_pos = current_pos-start_pos + delta_pos_pitch_compensation

        #rotate the rot and pos
        delta_pos, delta_rot = recorder_functions.transform_pose(self.R_BoxToKuka,delta_pos, delta_rot)

        #apply to calibrated start position
        self.kuka_new_pos = self.kuka_start_pos+delta_pos
        self.kuka_new_rot = self.kuka_start_rot+delta_rot

        #send the new position and rotation
        self.send_pose(self.kuka_new_pos,self.kuka_new_rot)

        # update the gui--------------------------------------------------------------------------------------
        yaw_setpoint = 0
        error_yaw = 0
        yaw_pterm = 0
        yaw_iterm = 0

        self.current_yaw_label .config(text=f"{yaw_setpoint:.3f} rad")
        self.yaw_setpoint_label.config(text=f"{yaw_setpoint:.3f} rad")
        self.yaw_error_label  .config(text=f"{error_yaw:.3f} rad")
        self.yaw_pterm_label  .config(text=f"{yaw_pterm:.3f}")
        self.yaw_iterm_label  .config(text=f"{yaw_iterm:.3f}")
        self.yaw_comp_label   .config(text=f"{delta_rot[1]:.3f}")

        self.current_pitch_label .config(text=f"{current_pitch:.3f} rad")
        self.pitch_setpoint_label.config(text=f"{pitch_setpoint:.3f} rad")
        self.pitch_error_label   .config(text=f"{error_pitch:.3f} rad")
        self.pitch_pterm_label   .config(text=f"{pitch_compensation_pterm:.3f}")
        self.pitch_iterm_label   .config(text=f"{pitch_compensation_iterm:.3f}")
        self.pitch_comp_label    .config(text=f"{pitch_compensation:.3f}")

        # record the parameters ---------------------------------------------------------------------------------
        if self.recording:
            # ───── pitch channels ─────────────────────────────
            self.pitch_setpoint_rec.append(pitch_setpoint)
            self.current_pitch_rec.append(current_pitch)
            self.pitch_error_rec.append(error_pitch)
            self.pitch_comp_rec.append(pitch_compensation)
            self.pitch_comp_iterm_rec.append(pitch_compensation_iterm)
            self.pitch_comp_pterm_rec.append(pitch_compensation_pterm)

            # ───── yaw channels (you already computed them above) ─
            self.current_yaw_rec.append(yaw_setpoint) 
            self.yaw_setpoint_rec.append(yaw_setpoint)
            self.yaw_error_rec.append(error_yaw)
            self.yaw_pterm_rec.append(yaw_pterm)
            self.yaw_iterm_rec.append(yaw_iterm)
            self.yaw_comp_rec.append(delta_rot[1])      # the compensation you sent
        return 

    def calibrate_kuka(self):
        """Store the current KUKA pose as the calibration reference."""
        self.kuka_start_pos,self.kuka_start_rot = self.get_current_pose()
        self.calibrate_kuka_button.config(text="KUKA Calibrated",style='CalibrateKukaDone.TButton')

    def set_new_velocity(self):
        """Read velocity input (Hz), convert to rad/s, and update motor speed setting."""

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
        self.motor_velocity_rad = -rad_per_sec_velocity #minus used since forward is different in my robot case
        print(f"motor velocity is set to: {hz_velocity} Hz")

    def compute_initial_world_xyz(self):
        """Compute the initial world-frame (X0, Y0, Z0) position of the UMR."""

        u, v = self.umr_1_center_x, self.umr_1_center_y
        
        # depth from cam2 measurement (back-projection)
        Z1 = (self.UMR_2_center_y - self.cy_cam2) * self.Z2[-1] / self.fy_cam2

        # back-project into cam1 coordinates
        X_rel = (u - self.cx_cam1) * self.Z1_initial / self.fx_cam1
        Y_rel = (v - self.cy_cam1) * self.Z1_initial / self.fy_cam1
        Z_rel = Z1

        # Shift into “box‐corner” world frame in X and Y
        X0 = X_rel - self.X_shift
        Y0 = -(Y_rel - self.Y_shift)

        # shift into box-corner world frame
        X0 = X_rel - self.X_shift
        Y0 = -(Y_rel - self.Y_shift)
        Z0 = -(Z_rel + self.Z_shift)

        return X0, Y0, Z0
    
    def update_frame(self):
        """Periodic GUI update: read cameras, update ROIs, record, plot, and run controller."""
        
        # stop if window closed
        if not self.window.winfo_exists():
            return
        
        # read the frames
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        # RECORDING PART----------------------------------------------------------------------- 
        if self.recording and ret1 and ret2: 
            self.N_frames += 1
            self.out1.write(frame1)
            self.out2.write(frame2)

            timestamp = time.time() - self.record_start_time
            self.timestamps.append(timestamp)
            self.X_3d_recording.append(self.X_3d[-1])
            self.Y_3d_recording.append(self.Y_3d[-1])
            self.Z_3d_recording.append(self.Z_3d[-1])
            
            if not self.controller_boolean: # if the controller is not active, we append zeros to the controller records
                #pitch channels
                self.pitch_setpoint_rec.append(0)
                self.current_pitch_rec.append(0)
                self.pitch_error_rec.append(0)
                self.pitch_comp_rec.append(0)
                self.pitch_comp_iterm_rec.append(0)
                self.pitch_comp_pterm_rec.append(0)

                #yaw channels
                self.current_yaw_rec.append(0)
                self.yaw_setpoint_rec.append(0)
                self.yaw_error_rec.append(0)
                self.yaw_pterm_rec.append(0)
                self.yaw_iterm_rec.append(0)
                self.yaw_comp_rec.append(0)

        # CAMERA 1 PART-----------------------------------------------------------------------    
        if ret1:
            frame1_display = frame1.copy() # Update the GUI with the frame --> first the frame is resized to fit in the GUI
            if self.UMR_1_roi is not None:
                # Draw the box around the UMR
                x, y, w, h = [int(v) for v in self.box_1_roi]
                cv2.rectangle(frame1_display, (x,y), (x+w,y+h), (255,0,0), 4)

                # Update the tracker 
                self.UMR_1_bounding_box, self.UMR_1_roi,self.UMR_1_angle_measured, self.UMR_1_center_x, self.UMR_1_center_y = recorder_functions.update_roi_center(frame1, self.UMR_1_roi)

                # Draw the bounding box around the umr
                x, y, w, h = [int(v) for v in self.UMR_1_roi]
                #cv2.rectangle(frame1_display, (x,y), (x+w,y+h), (0,0,255), 4) #This can draw the ROI around the umr, but it is not needed
                cv2.drawContours(frame1_display, [self.UMR_1_bounding_box + (x, y)], 0, (0, 0, 255), 2) # Draw the bounding box around the umr 

            frame1_resized = cv2.resize(frame1_display, (576, 324), interpolation=cv2.INTER_LINEAR)    
            frame_rgb1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
            img1 = ImageTk.PhotoImage(Image.fromarray(frame_rgb1))
            self.video_label1.imgtk = img1
            self.video_label1.config(image=img1)

        # CAMERA 2 PART------------------------------------------------------------------------
        if ret2:
            frame2_display = frame2.copy() # Update the GUI with the frame --> first the frame is resized to fit in the GUI 
            if self.UMR_2_roi is not None:
                # Draw the box around the UMR
                x, y, w, h = [int(v) for v in self.box_2_roi]
                cv2.rectangle(frame2_display, (x,y), (x+w,y+h), (255,0,0), 4)

                # Update the tracker 
                self.UMR_2_bounding_box,self.UMR_2_roi,self.UMR_2_angle_measured, self.UMR_2_center_x, self.UMR_2_center_y = recorder_functions.update_roi_center(frame2, self.UMR_2_roi)
                
                # draw the bounding box around the umr
                x, y, w, h = [int(v) for v in self.UMR_2_roi]
                #cv2.rectangle(frame2_display, (x,y), (x+w,y+h), (0,0,255), 4) #This can draw the ROI around the umr, but it is not needed
                cv2.drawContours(frame2_display, [self.UMR_2_bounding_box + (x, y)], 0, (0, 0, 255), 2) # Draw the bounding box around the umr   

            frame2_resized = cv2.resize(frame2_display, (576, 324), interpolation=cv2.INTER_LINEAR)    
            frame_rgb2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
            img2 = ImageTk.PhotoImage(Image.fromarray(frame_rgb2))
            self.video_label2.imgtk = img2
            self.video_label2.config(image=img2)

        # RECONSTRUCTION PART-------------------------------------------------------------------
        if self.reconstruction_boolean is True:
            # Update the reconstructed pose and angle in the GUI based on the UMRs and box positions
            self.update_trajectory()
            if len(self.X_3d) % 5 == 0:  # only redraw every 5th frame
                self.update_trajectory_plot()

            # Get the angle and update the labels
            self.angle_1, self.angle_2 = self.angle_filter(self.UMR_1_angle_measured, self.UMR_2_angle_measured)
            self.angle1_label.config(text=f"Angle 1: {math.degrees(self.angle_1):.2f}°")
            self.angle2_label.config(text=f"Angle 2: {math.degrees(self.angle_2):.2f}°")

        # CONTROLLER PART-----------------------------------------------------------------------
        if self.controller_boolean is True:
            if self.reconstruction_boolean is False:
                print("First turn the reconstructor on")
                return
            time_controller = time.perf_counter()
            self.controller(time_controller) # or self.straight_controller(time_controller) for straight controller

        self.window.after(10, self.update_frame)

    def on_closing(self):
        """Handle safe shutdown when closing the application window."""

        # stop recording safely
        if getattr(self, "recording", False):
            try:
                if hasattr(self, "out1"): 
                    self.out1.release()
                if hasattr(self, "out2"): 
                    self.out2.release()
            except Exception as e:
                print(f"[WARN] Failed to release video writers: {e}")

        # release cameras
        try:
            if hasattr(self, "cap1"): 
                self.cap1.release()
            if hasattr(self, "cap2"): 
                self.cap2.release()
        except Exception as e:
            print(f"[WARN] Failed to release cameras: {e}")

        # close main window
        if hasattr(self, "window"):
            self.window.destroy()

        # shut down publisher
        try:
            self.kuka_python.shutdown_publisher()
        except Exception as e:
            print(f"[WARN] Failed to shut down KUKA publisher: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClosedLoopRecorder(root)
    root.mainloop()


    #TODO: update controller parameters JSON file
    #TODO: nadenken of we pitch ook een direction willen geven al vantevoren
    #TODO: heading plotten
    #TODO: distance alleen y