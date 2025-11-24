import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, CompressedImage

import cv2
import cv_bridge
import numpy as np
import os
import time


# CONSTANTS
INIT                           = "INIT"
LOAD_SLAM_MAP                  = "LOAD_SLAM_MAP"
LOCALIZE_WITH_FILTER           = "LOCALIZE_WITH_FILTER"
WALL_FOLLOW                    = "WALL_FOLLOW"
OBSERVE_OBJECT                 = "OBSERVE_OBJECT"
WAIT_FOR_TARGET                = "WAIT_FOR_TARGET"
PLAN_PATH_TO_OBJECT            = "PLAN_PATH_TO_OBJECT"
NAVIGATE_TO_OBJECT             = "NAVIGATE_TO_OBJECT"
ALIGN_WITH_OBJECT              = "ALIGN_WITH_OBJECT"
PICK_UP_OBJECT                 = "PICK_UP_OBJECT"
PLAN_PATH_TO_BIN               = "PLAN_PATH_TO_BIN"
NAVIGATE_TO_BIN                = "NAVIGATE_TO_BIN"
DROP_OBJECT                    = "DROP_OBJECT"
TASK_COMPLETE                  = "TASK_COMPLETE"


class ObjectCollector(Node):

    def __init__(self):
        super().__init__('maze_object_collector')
        
        ros_domain_id = os.getenv("ROS_DOMAIN_ID")
        self.get_logger().info(f"ROS_DOMAIN_ID: {ros_domain_id}")

        self.bridge = cv_bridge.CvBridge()

        # State machine
        self.robot_state = INIT
        self.map_loaded = False
        self.map = None

        # Object tracking
        self.object_positions = {"blue": None, "green": None, "pink": None}
        self.objects_found    = {"blue": False, "green": False, "pink": False}

        # Subscribers
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.get_map_callback,
            10
        )

        self.create_subscription(
            LaserScan,
            f"/tb{ros_domain_id}/scan",
            self.scan_callback,
            10
        )

        self.create_subscription(
            CompressedImage,
            f"/tb{ros_domain_id}/oakd/rgb/preview/image_raw/compressed",
            self.image_callback,
            10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            Twist, 
            f"/tb{ros_domain_id}/cmd_vel",
            10
        )

        # State machine timer 
        self.create_timer(0.1, self.state_machine_step)

        self.get_logger().info("ObjectCollector Node Initialized")


    # Map loading
    def get_map_callback(self, msg: OccupancyGrid):
        """Receive a SLAM map from /map and store it."""
        if not self.map_loaded:
            self.map = msg
            self.map_loaded = True
            self.get_logger().info("SLAM map received from /map.")


    # Scan and image callbacks
    def scan_callback(self, msg):
        pass

    def image_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        pass



    # State machine
    def state_machine_step(self):

        if self.robot_state == INIT:
            self.handle_init()

        elif self.robot_state == LOAD_SLAM_MAP:
            self.handle_load_slam_map()

        elif self.robot_state == LOCALIZE_WITH_FILTER:
            self.handle_particle_filter_localization()

        elif self.robot_state == WALL_FOLLOW:
            self.handle_wall_follow()

        elif self.robot_state == OBSERVE_OBJECT:
            self.handle_observe_object()

        elif self.robot_state == WAIT_FOR_TARGET:
            self.handle_wait_for_target()

        elif self.robot_state == PLAN_PATH_TO_OBJECT:
            self.handle_plan_to_object()

        elif self.robot_state == NAVIGATE_TO_OBJECT:
            self.handle_navigate_to_object()

        elif self.robot_state == ALIGN_WITH_OBJECT:
            self.handle_align_with_object()

        elif self.robot_state == PICK_UP_OBJECT:
            self.handle_pick_up_object()

        elif self.robot_state == PLAN_PATH_TO_BIN:
            self.handle_plan_to_bin()

        elif self.robot_state == NAVIGATE_TO_BIN:
            self.handle_navigate_to_bin()

        elif self.robot_state == DROP_OBJECT:
            self.handle_drop_object()

        elif self.robot_state == TASK_COMPLETE:
            self.handle_task_complete()



    # State handlers

    def handle_init(self):
        self.get_logger().info("INIT → LOAD_SLAM_MAP")
        self.robot_state = LOAD_SLAM_MAP


    def handle_load_slam_map(self):
        """Wait until the /map subscription delivers the SLAM map."""
        if not self.map_loaded:
            self.get_logger().info("Waiting for SLAM map from /map...")
            return

        self.get_logger().info("SLAM map loaded. Proceeding to localization.")
        self.robot_state = LOCALIZE_WITH_FILTER


    def handle_particle_filter_localization(self):
        self.get_logger().info("Running particle filter localization...")
        # TODO: Integrate PF results
        localization_done = False
        if localization_done:
            self.robot_state = WALL_FOLLOW


    def handle_wall_follow(self):
        self.get_logger().info("Wall following...")
        # TODO: Wall follower implementation
        pass


    def handle_observe_object(self):
        self.get_logger().info("Observing object...")
        pass


    def handle_wait_for_target(self):
        if self.target_object is not None:
            self.robot_state = PLAN_PATH_TO_OBJECT


    def handle_plan_to_object(self):
        self.current_path = []
        self.robot_state = NAVIGATE_TO_OBJECT


    def handle_navigate_to_object(self):
        arrived = False
        if arrived:
            self.robot_state = ALIGN_WITH_OBJECT


    def handle_align_with_object(self):
        refined = True
        if refined:
            self.robot_state = PICK_UP_OBJECT


    def handle_pick_up_object(self):
        self.has_object = True
        self.robot_state = PLAN_PATH_TO_BIN


    def handle_plan_to_bin(self):
        self.current_path = []
        self.robot_state = NAVIGATE_TO_BIN


    def handle_navigate_to_bin(self):
        arrived = False
        if arrived:
            self.robot_state = DROP_OBJECT


    def handle_drop_object(self):
        self.has_object = False
        self.robot_state = TASK_COMPLETE


    def handle_task_complete(self):
        self.get_logger().info("Task complete.")
        self.publish_velocity(0.0, 0.0)



    # helper functions here 
    def publish_velocity(self, lin, ang):
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)



# maib
def main(args=None):
    rclpy.init(args=args)
    node = ObjectCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
