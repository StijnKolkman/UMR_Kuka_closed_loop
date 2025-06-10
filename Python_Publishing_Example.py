from kuka_python_node.kuka_node import start_kuka_node

#setup
kuka_python = start_kuka_node()

#sending a pose
sending_pose = (0.0,0.0,0.0,0.0,0.0,0.0) #(x,y,x,a,b,c) in mm and degrees
kuka_python.publish_pose(sending_pose) #Publish the pose to the ROS topic.

#get the current pose
kuka_incoming_pose = kuka_python.get_position() #get the current pose of the kuka in m and radians

#set the maxon velocity
motor_velocity_rad = 1 #motor velocity in rad
kuka_python.set_motor_speed(motor_velocity_rad)