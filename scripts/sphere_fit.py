#!/usr/bin/env python3

import rospy
import numpy as np
from robot_vision_lectures.msg import XYZarray, SphereParams
from geometry_msgs.msg import Point

class SphereFitNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('sphere_fit_node', anonymous=True)
        rospy.loginfo("Sphere Fit Node initialized")
        
        # Subscriber for receiving XYZ points
        self.subscriber = rospy.Subscriber('/xyz_cropped_ball', XYZarray, self.xyz_callback, queue_size=1)
        
        # Publisher for sending sphere parameters
        self.publisher = rospy.Publisher('/sphere_params', SphereParams, queue_size=1)
        
        # Set the publishing rate
        self.rate = rospy.Rate(10)  # 10 Hz
        
        # Variable to store received XYZ points
        self.xyz_points = None

    # Callback function for processing received XYZ points
    def xyz_callback(self, msg):
        rospy.loginfo("Received XYZ points")
        self.xyz_points = np.array([[point.x, point.y, point.z] for point in msg.points])

    # Function to fit a sphere to the received XYZ points
    def fit_sphere(self):
        # Check if sufficient XYZ points are available
        if self.xyz_points is None or len(self.xyz_points) < 4:
            rospy.logwarn("Insufficient XYZ points received.")
            return None

        # Prepare matrices for solving the least squares problem
        A = np.column_stack((2*self.xyz_points, np.ones(len(self.xyz_points))))
        B = np.sum(self.xyz_points**2, axis=1)

        # Solve the least squares problem to obtain sphere parameters
        P = np.linalg.lstsq(A, B, rcond=None)[0]

        # Calculate sphere center coordinates and radius
        x_c, y_c, z_c = P[0], P[1], P[2]
        radius = np.sqrt(x_c**2 + y_c**2 + z_c**2 + P[3])

        return x_c, y_c, z_c, radius

    # Main function to run the node
    def run(self):
        rospy.loginfo("Sphere Fit Node is running")
        while not rospy.is_shutdown():
            rospy.loginfo("Fitting sphere")
            sphere_params_msg = SphereParams()
            
            # Fit a sphere to the received XYZ points
            params = self.fit_sphere()

            if params is None:
                rospy.logwarn("Sphere fitting failed")
                continue

            rospy.loginfo("Sphere parameters: %s", params)

            # Populate the SphereParams message with calculated parameters
            sphere_params_msg.xc, sphere_params_msg.yc, sphere_params_msg.zc, sphere_params_msg.radius = params

            # Publish the SphereParams message
            self.publisher.publish(sphere_params_msg)
        
            self.rate.sleep()

if __name__ == '__main__':
    try:
        # Initialize and run the SphereFitNode
        sphere_fit_node = SphereFitNode()
        sphere_fit_node.run()
    except rospy.ROSInterruptException:
        pass
