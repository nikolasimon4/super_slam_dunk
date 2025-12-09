import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion, PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import cv2
import cv_bridge
import numpy as np
import os
import time
import math
import heapq


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

# Test states
TEST_WAIT_FOR_READY            = "TEST_WAIT_FOR_READY"
TEST_PROMPT_TARGET             = "TEST_PROMPT_TARGET"
TEST_PLAN_AND_LOG              = "TEST_PLAN_AND_LOG"
FOLLOW_PATH                    = "FOLLOW_PATH"

# Whenn true, skips wall following, prompts for target)
TEST_MODE = True


class ObjectCollector(Node):

    def __init__(self):
        super().__init__('maze_object_collector')

        # Pull turtlebot number and pad with leading zero if needed
        unformatted = os.getenv('ROS_DOMAIN_ID')
        ros_domain_id = f'{int(unformatted):02d}'
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')

        self.bridge = cv_bridge.CvBridge()
        
        #Wall following
        self.desired_distance = .25
        self.recent_scan = None
        # State machine
        self.robot_state = INIT
        self.map_loaded = False
        self.map = None
        self.obstacle_map = None

        # Object tracking
        self.object_positions = {"blue": None, "green": None, "pink": None}
        self.objects_found    = {"blue": False, "green": False, "pink": False}
        
        # Runtime state variables
        self.target_object = None
        self.current_path = []
        self.has_object = False
        self.robot_pose = None
        self.safe_angle = -math.pi / 2
        self.closest_object_forward = None
        self.closest_object_left = None
        
        # Test mode flag
        self.test_prompted = False
        self.clicked_goal = None
        
        # Path following parameters
        self.path_index = 0                    # Current waypoint index
        self.waypoint_tolerance = 0.20         # How close to waypoint before moving to next (meters)
        self.goal_tolerance = 0.15             # How close to final goal (meters)
        self.linear_speed = 0.12               # Forward speed
        self.angular_speed = 0.4               # Max turning speed
        self.angle_tolerance = 0.4             # Angle error before moving forward (radians)
        
        qos_profile = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscribers
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.get_map_callback,
            qos_profile
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
        self.create_subscription(
            PoseStamped,
            f"/tb{ros_domain_id}/estimated_robot_pose",
            self.pose_callback,
            10
        )
        
        # Subscribe to RViz clicked point for testing A* paths
        self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.clicked_point_callback,
            10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            Twist, 
            f"/tb{ros_domain_id}/cmd_vel",
            10
        )
        
        # Visualization publishers
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        self.path_marker_pub = self.create_publisher(Marker, "/planned_path", 10)
        self.path_pub = self.create_publisher(Path, "/astar_path", 10)

        # State machine timer 
        self.create_timer(0.1, self.state_machine_step)

        self.get_logger().info("ObjectCollector Node Initialized")

    def pose_callback(self, pose: PoseStamped):
        """
        Callback for pose subscription
        """ 
        self.robot_pose = pose

    def clicked_point_callback(self, msg: PointStamped):
        """
        Callback for RViz 'Publish Point' tool clicks.
        Click anywhere on the map to test A* path planning.
        """
        goal_x = msg.point.x
        goal_y = msg.point.y
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("CLICKED POINT RECEIVED")
        self.get_logger().info(f"Goal: ({goal_x:.3f}, {goal_y:.3f})")
        
        # Check if map is loaded
        if not self.map_loaded:
            self.get_logger().error("No map loaded yet!")
            return
        
        # Require robot pose for path planning
        if self.robot_pose is None:
            self.get_logger().error("No robot pose yet! Make sure particle filter is running.")
            self.get_logger().info("=" * 60)
            return
        
        start_x = self.robot_pose.pose.position.x
        start_y = self.robot_pose.pose.position.y
        
        self.get_logger().info(f"Start: ({start_x:.3f}, {start_y:.3f})")
        
        # Check if goal is valid
        if not self.is_valid_goal(goal_x, goal_y):
            self.get_logger().warn("Goal is in obstacle or out of bounds!")
        
        # Plan path
        path = self.plan_path((start_x, start_y), (goal_x, goal_y))
        
        if not path:
            self.get_logger().error("NO PATH FOUND!")
            self.get_logger().info("=" * 60)
            return
        
        # Log waypoints (sample if too many)
        self.get_logger().info(f"PATH FOUND: {len(path)} waypoints")
        self.get_logger().info("-" * 40)
        
        # Show first 5, last 5, and every 10th in between
        for i, (x, y) in enumerate(path):
            if i < 5 or i >= len(path) - 5 or i % 10 == 0:
                self.get_logger().info(f"  [{i:3d}] ({x:7.3f}, {y:7.3f})")
            elif i == 5:
                self.get_logger().info(f"  ... ({len(path) - 10} more waypoints) ...")
        
        # Calculate total distance
        total_dist = sum(
            math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
            for i in range(1, len(path))
        )
        
        self.get_logger().info("-" * 40)
        self.get_logger().info(f"Total distance: {total_dist:.2f} meters")
        self.get_logger().info("STARTING PATH FOLLOWING...")
        self.get_logger().info("=" * 60)
        
        # Store path and start following
        self.clicked_goal = (goal_x, goal_y)
        self.current_path = path
        self.path_index = 0
        
        # Publish visualization to RViz
        self.publish_goal_marker(goal_x, goal_y)
        self.publish_path_visualization(path)
        
        self.robot_state = FOLLOW_PATH

    # Map loading
    def get_map_callback(self, msg: OccupancyGrid):
        """Receive a SLAM map from /map and store it."""
        if not self.map_loaded:
            self.map = msg
            self.map_loaded = True
            self.get_logger().info("SLAM map received from /map.")

    # Scan and image callbacks
    def scan_callback(self, msg):
        self.closest_object_forward = self.find_closest_object_angle(msg,-math.pi/8,math.pi/8)
        self.closest_object_left = self.find_closest_object_angle(msg,-3 * math.pi / 4, -1 * math.pi/4)
        self.recent_scan = msg
        # Only run wall follow in those specific states (not during path following)
        if self.robot_state == LOCALIZE_WITH_FILTER or self.robot_state == WALL_FOLLOW:
            self.wall_follow_publish()

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

        # Test states
        elif self.robot_state == TEST_WAIT_FOR_READY:
            self.handle_test_wait_for_ready()

        elif self.robot_state == FOLLOW_PATH:
            self.handle_follow_path()

    # State handlers

    def handle_init(self):
        self.get_logger().info("INIT → LOAD_SLAM_MAP")
        self.robot_state = LOAD_SLAM_MAP

    def handle_load_slam_map(self):
        """Wait until the /map subscription delivers the SLAM map."""
        if not self.map_loaded:
            self.get_logger().info("Waiting for SLAM map from /map...")
            return

        self.get_logger().info("SLAM map loaded.")
        
        if TEST_MODE:
            self.get_logger().info("TEST MODE: Going to test flow...")
            self.robot_state = TEST_WAIT_FOR_READY
        else:
            self.get_logger().info("Proceeding to localization.")
            self.robot_state = LOCALIZE_WITH_FILTER

    def handle_particle_filter_localization(self):
        self.get_logger().info("Running particle filter localization...")
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
        """Plan path from current robot pose to target object."""
        if self.robot_pose is None or self.target_object is None:
            self.get_logger().error("Cannot plan: missing robot_pose or target_object")
            return
        
        goal = self.object_positions.get(self.target_object)
        if goal is None:
            self.get_logger().error(f"No position for target: {self.target_object}")
            return
        
        self.current_path = self.plan_path(self.robot_pose, goal)
        
        if self.current_path:
            self.get_logger().info(f"Path to {self.target_object}: {len(self.current_path)} waypoints")
            self.robot_state = NAVIGATE_TO_OBJECT
        else:
            self.get_logger().error("Failed to plan path to object")
            # TODO: handle failure (eg go back to exploration)

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
        """Plan path from current robot pose to the appropriate bin."""
        if self.robot_pose is None:
            self.get_logger().error("Cannot plan: missing robot_pose")
            return
        
        # TODO: determine bin position based on object color
        bin_position = None  # like self.bin_positions["trash"] or ["recycling"]
        
        if bin_position is None:
            self.get_logger().warning("Bin position not set, skipping to navigate")
            self.current_path = []
            self.robot_state = NAVIGATE_TO_BIN
            return
        
        self.current_path = self.plan_path(self.robot_pose, bin_position)
        
        if self.current_path:
            self.get_logger().info(f"Path to bin: {len(self.current_path)} waypoints")
            self.robot_state = NAVIGATE_TO_BIN
        else:
            self.get_logger().error("Failed to plan path to bin")

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

    # =========================================================================
    # TEST MODE HANDLERS
    # =========================================================================

    def handle_test_wait_for_ready(self):
        """Wait for map, then allow click-to-path testing."""
        if not self.map_loaded:
            self.get_logger().info("TEST: Waiting for map...", throttle_duration_sec=2.0)
            return
        
        if not self.test_prompted:
            self.test_prompted = True
            self.get_logger().info("=" * 60)
            self.get_logger().info("CLICK-TO-PATH TEST MODE READY")
            self.get_logger().info("=" * 60)
            self.get_logger().info("In RViz: Use 'Publish Point' tool to click on the map")
            self.get_logger().info("Robot pose will be used when available")
            self.get_logger().info("=" * 60)
        
        # Stay in this state - clicks are handled by clicked_point_callback

    def handle_test_plan_and_log(self):
        """Unused - kept for compatibility."""
        pass

    def handle_follow_path(self):
        """
        Follow the planned path waypoint by waypoint.
        Uses simple proportional control for turning and constant speed for forward motion.
        """
        # Safety checks
        if self.robot_pose is None:
            self.get_logger().warn("Lost robot pose during path following!", throttle_duration_sec=1.0)
            self.publish_velocity(0.0, 0.0)
            return
        
        if not self.current_path or self.path_index >= len(self.current_path):
            self.get_logger().info("Path complete or empty!")
            self.publish_velocity(0.0, 0.0)
            self.robot_state = TEST_WAIT_FOR_READY
            return
        
        # Get current position and target waypoint
        robot_x = self.robot_pose.pose.position.x
        robot_y = self.robot_pose.pose.position.y
        robot_yaw = self.get_yaw_from_pose(self.robot_pose.pose)
        
        target_x, target_y = self.current_path[self.path_index]
        
        # Calculate distance and angle to waypoint
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx * dx + dy * dy)
        angle_to_target = math.atan2(dy, dx)
        
        # Angle error (wrapped to [-pi, pi])
        angle_error = self.wrap_angle(angle_to_target - robot_yaw)
        
        # Check if we've reached the current waypoint
        is_final_waypoint = (self.path_index == len(self.current_path) - 1)
        tolerance = self.goal_tolerance if is_final_waypoint else self.waypoint_tolerance
        
        if distance < tolerance:
            if is_final_waypoint:
                # Reached final goal!
                self.get_logger().info("=" * 40)
                self.get_logger().info("GOAL REACHED!")
                self.get_logger().info("=" * 40)
                self.publish_velocity(0.0, 0.0)
                self.clear_path_visualization()
                self.current_path = []
                self.robot_state = TEST_WAIT_FOR_READY
                return
            else:
                # Move to next waypoint
                self.path_index += 1
                self.get_logger().info(f"Waypoint {self.path_index}/{len(self.current_path)} reached")
                return
        
        # Debug logging (throttled)
        self.get_logger().info(
            f"[{self.path_index}/{len(self.current_path)}] "
            f"pos=({robot_x:.2f},{robot_y:.2f}) yaw={math.degrees(robot_yaw):.1f}° "
            f"target=({target_x:.2f},{target_y:.2f}) "
            f"dist={distance:.2f}m angle_err={math.degrees(angle_error):.1f}°",
            throttle_duration_sec=0.5
        )
        
        # Control logic: Always move forward, steer proportionally
        # Proportional steering (gentler gain)
        angular_vel = max(-self.angular_speed, min(self.angular_speed, angle_error * 0.5))
        
        # Always move forward unless angle is very wrong (> 90 degrees)
        ## > 90 degrees - need to turn more
        if abs(angle_error) > 1.57:
            linear_vel = 0.0
            angular_vel = self.angular_speed if angle_error > 0 else -self.angular_speed
        ## > 45 degrees - slow down
        elif abs(angle_error) > 0.8:
            linear_vel = self.linear_speed * 0.3
        ## Roughly pointed at target - drive forward
        else:
            linear_vel = self.linear_speed
        
        self.publish_velocity(linear_vel, angular_vel)

    def get_yaw_from_pose(self, pose):
        """Extract yaw angle from a Pose quaternion."""
        q = pose.orientation
        # Yaw from quaternion: atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def publish_goal_marker(self, x, y):
        """Publish a sphere marker at the goal position."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        # Green color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 0  # Persistent
        
        self.goal_marker_pub.publish(marker)

    def publish_path_visualization(self, path):
        """Publish the planned path as a line strip marker and nav_msgs/Path."""
        if not path:
            return
        
        # Publish as LINE_STRIP marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.05  # Line width
        
        # Cyan/blue color
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        marker.lifetime.sec = 0  # Persistent
        
        # Add all path points
        for x, y in path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.05
            marker.points.append(p)
        
        self.path_marker_pub.publish(marker)
        
        # Also publish as nav_msgs/Path (useful for other tools)
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def clear_path_visualization(self):
        """Clear the path and goal markers."""
        # Clear goal
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.action = Marker.DELETE
        self.goal_marker_pub.publish(marker)
        
        # Clear path
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.action = Marker.DELETE
        self.path_marker_pub.publish(marker)

    # Helper functions
 
    def publish_velocity(self, lin, ang):
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)

    def build_obstacle_map(self, inflation_radius_meters=0.18):
        """
        Build a binary obstacle map for A* pathfinding with inflated obstacles.
        
        Args:
            inflation_radius_meters: How much to inflate obstacles (robot radius + safety margin)
                                    Default 0.25m = ~0.17m robot + 0.08m safety
        """
        if self.map is None:
            self.get_logger().error("No map available to build obstacle map.")
            return

        width = self.map.info.width
        height = self.map.info.height
        resolution = self.map.info.resolution
        data = self.map.data

        # Create 2D numpy array from occupancy grid
        occupancy_2d = np.array(data, dtype=np.int8).reshape((height, width))
        
        # Create binary obstacle map (0 = free, 1 = obstacle)
        # Mark occupied (>50) and unknown (<0) as obstacles
        obstacle_binary = np.where((occupancy_2d > 50) | (occupancy_2d < 0), 1, 0).astype(np.uint8)
        
        # Calculate inflation radius in grid cells
        inflation_cells = int(math.ceil(inflation_radius_meters / resolution))
        
        self.get_logger().info(f"Inflating obstacles by {inflation_radius_meters}m ({inflation_cells} cells)")
        
        # Create circular structuring element (kernel) for dilation
        kernel_size = 2 * inflation_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate obstacles (inflate them)
        inflated_2d = cv2.dilate(obstacle_binary, kernel, iterations=1)
        
        # Convert back to 1D list with occupancy values (0 or 100)
        obstacle_map = (inflated_2d.flatten() * 100).tolist()
        
        self.obstacle_map = obstacle_map
        
        # Log statistics
        original_obstacles = np.sum(obstacle_binary)
        inflated_obstacles = np.sum(inflated_2d)
        self.get_logger().info(f"Obstacle map built: {original_obstacles} → {inflated_obstacles} cells "
                              f"({100*inflated_obstacles/(width*height):.1f}% of map)")

    def get_neighbors_1d(self, idx):
        """Return indexes of 8 neighbors."""
        width = self.map.info.width
        height = self.map.info.height

        neighbors = []
        x = idx % width
        y = idx // width

        # (dx, dy) directions
        deltas = [
            (-1,  0), (1,  0),      # left, right
            (0, -1), (0,  1),       # up, down
            (-1, -1), (1, -1),      # diag up-left, up-right
            (-1,  1), (1,  1)       # diag down-left, down-right
        ]

        for dx, dy in deltas:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append(ny * width + nx)

        return neighbors


    def astar_indices(self, start_idx, goal_idx):
        """
        A* search, returns path in indexes on map.
        """
        width  = self.map.info.width
        height = self.map.info.height

        # Make sure obstacle map is built
        if self.obstacle_map is None:
            self.build_obstacle_map()

        # Bounds check
        if not (0 <= start_idx < width * height) or not (0 <= goal_idx < width * height):
            self.get_logger().error("Start or goal index out of bounds.")
            return []

        # Collision check
        if self.obstacle_map[start_idx] == 100:
            self.get_logger().error("Start index is an obstacle.")
            return []
        if self.obstacle_map[goal_idx] == 100:
            self.get_logger().error("Goal index is an obstacle.")
            return []

        # A* structures
        open_heap = []
        heap_counter = 0
        heapq.heappush(open_heap, (0.0, heap_counter, start_idx))

        g_cost = {start_idx: 0.0}
        came_from = {}

        # Storage for last move
        last_move = {start_idx: (0, 0)}

        closed = set()

        sqrt2 = math.sqrt(2.0)

        def heuristic(i):
            """Heuristic for distance finding"""
            x1 = i % width
            y1 = i // width
            x2 = goal_idx % width
            y2 = goal_idx // width 
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            return (dx + dy)

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == goal_idx:
                return self.reconstruct_indices_path(came_from, goal_idx)

            cx = current % width
            cy = current // width

            for nb in self.get_neighbors_1d(current):
                if self.obstacle_map[nb] == 100:
                    continue

                nx = nb % width
                ny = nb // width

                dx = nx - cx
                dy = ny - cy

                # Detect diagonal move
                diag = (abs(dx) == 1 and abs(dy) == 1)

                # Prevent corner-cutting
                if diag:
                    mid1_x, mid1_y = cx + dx, cy      
                    mid2_x, mid2_y = cx, cy + dy 
                    mid1_idx = mid1_y * width + mid1_x
                    mid2_idx = mid2_y * width + mid2_x

                    if (self.obstacle_map[mid1_idx] == 100 or
                        self.obstacle_map[mid2_idx] == 100):
                        continue

                base_cost = sqrt2 if diag else 1.0

                # Direction change penalty
                prev_dx, prev_dy = last_move.get(current, (0, 0))
                turn_penalty = 0.0
                if (prev_dx, prev_dy) != (0, 0) and (dx, dy) != (prev_dx, prev_dy):
                    turn_penalty = 0.1

                new_g = g_cost[current] + base_cost + turn_penalty

                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    came_from[nb] = current
                    last_move[nb] = (dx, dy)

                    f = new_g + heuristic(nb)
                    heap_counter += 1
                    heapq.heappush(open_heap, (f, heap_counter, nb))

        self.get_logger().warning("A* found no path.")
        return []

    def reconstruct_indices_path(self, came_from, goal_idx):
        """Return a list of 1D indices representing the path."""
        path = []
        cur = goal_idx
        while cur in came_from:
            path.append(cur)
            cur = came_from[cur]
        path.append(cur)
        path.reverse()
        return path

    # =========================================================================
    # Coordinate Conversion Helpers
    # =========================================================================

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.
        
        Args:
            x: X position in meters (map frame)
            y: Y position in meters (map frame)
            
        Returns:
            (row, col) tuple, or None if out of bounds
        """
        if self.map is None:
            return None
        
        res = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        
        col = int(math.floor((x - origin_x) / res))
        row = int(math.floor((y - origin_y) / res))
        
        if row < 0 or col < 0 or row >= self.map.info.height or col >= self.map.info.width:
            return None
        
        return (row, col)

    def grid_to_world(self, row, col):
        """
        Convert grid indices to world coordinates (center of cell).
        
        Args:
            row: Grid row index
            col: Grid column index
            
        Returns:
            (x, y) tuple in meters (map frame)
        """
        if self.map is None:
            return None
        
        res = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        
        # +0.5 to get center of cell
        x = origin_x + (col + 0.5) * res
        y = origin_y + (row + 0.5) * res
        
        return (x, y)

    def grid_to_index(self, row, col):
        """Convert (row, col) to 1D index."""
        return row * self.map.info.width + col

    def index_to_grid(self, idx):
        """Convert 1D index to (row, col)."""
        width = self.map.info.width
        return (idx // width, idx % width)

    # =========================================================================
    # Path Planning Interface
    # =========================================================================

    def plan_path(self, start, goal):
        """
        Plan a path from start to goal using A*.
                
        Args:
            start: (x, y) tuple in meters, or Pose object with .position.x/.y
            goal:  (x, y) tuple in meters, or Pose object with .position.x/.y
            
        Returnss:
            List of (x, y) waypoints in meters, or empty list if no path found.
        """
        if self.map is None:
            self.get_logger().error("Cannot plan path: no map loaded.")
            return []

        # Extract coordinates from Pose objects if needed
        if hasattr(start, 'position'):
            start = (start.position.x, start.position.y)
        if hasattr(goal, 'position'):
            goal = (goal.position.x, goal.position.y)

        # Convert world coords to grid
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])

        if start_grid is None:
            self.get_logger().error(f"Start position {start} is out of map bounds.")
            return []
        if goal_grid is None:
            self.get_logger().error(f"Goal position {goal} is out of map bounds.")
            return []

        # Convert to 1D indices
        start_idx = self.grid_to_index(start_grid[0], start_grid[1])
        goal_idx = self.grid_to_index(goal_grid[0], goal_grid[1])

        # Run A*
        path_indices = self.astar_indices(start_idx, goal_idx)

        if not path_indices:
            return []

        # Convert path back to world coordinates
        path_world = []
        for idx in path_indices:
            row, col = self.index_to_grid(idx)
            world_pos = self.grid_to_world(row, col)
            if world_pos:
                path_world.append(world_pos)

        self.get_logger().info(f"Path planned: {len(path_world)} waypoints")
        return path_world
    
    def is_valid_goal(self, x, y):
        """
        Check if a goal position is reachable (not in obstacle).
        
        Args:
            x, y: Position in meters
            
        Returns:
            True if position is valid for path planning
        """
        if self.map is None:
            return False
        
        if self.obstacle_map is None:
            self.build_obstacle_map()
        
        grid = self.world_to_grid(x, y)
        if grid is None:
            return False
        
        idx = self.grid_to_index(grid[0], grid[1])
        return self.obstacle_map[idx] != 100
    
    ### WALL FOLLOWER HELPERS AND CODE
    def find_closest_object_angle(self, scan: LaserScan, low: float, high: float) -> tuple[float, float]:
        """
        Finds closest object distance between low and high angles (with 0 being forward), given laser scan data 
        """
        ANG_ADJ = -math.pi/2
        # List of distances
        ranges = scan.ranges

        # Min/max angle from scan
        angle_min = scan.angle_min
        angle_max = scan.angle_max
        
        # Min/max dist from scan
        range_min = scan.range_min
        range_max = scan.range_max

        # Default values (not great edge case handling, but this would be a prob with sensor)
        min_dist_angle = 0.0
        min_dist = range_max

        # Looping over each distance in the sensor
        for index, dist in enumerate(ranges):
            # Thrown out if the range is too large or small (eg invalid) (done according to laserscan documentation)
            if dist > range_max or dist < range_min:
                continue

            # Adjust angle so it is from the point of reference of forward being 0
            zeroed_angle = (angle_min + index * scan.angle_increment - ANG_ADJ)
            
            # If adjustment happens we loop back around so that the actual angle we output is within range [-pi,pi]
            if min_dist_angle < angle_min:
                min_dist_angle = angle_max - angle_min + min_dist_angle 
            if min_dist_angle > angle_max:
                min_dist_angle = angle_min - angle_max + min_dist_angle
            
            # Continuing if the scan is for a distance not in front 
            if zeroed_angle > high or zeroed_angle < low:
                continue 
            # If this is a new closest object, set the min_distance to that and the angle to the current angle
            if dist < min_dist:
                min_dist = dist
                min_dist_angle = zeroed_angle

        
        return (min_dist_angle, min_dist)

    def wall_follow_publish(self):
        
        if(self.closest_object_forward):
            lin = 0
            ang = 0
            if(self.closest_object_forward[1] < self.desired_distance + .05):
                self.get_logger().error(f"PUBLISHING linear: {lin} ang {ang}, FORWARD")
                ang = 1.5
                lin = 0
            else:
                ang = (self.find_closest_object_angle(self.recent_scan, -math.pi/2 - math.pi /2 + math.pi/50, -math.pi/2 + math.pi/2 - math.pi/50)[0] - self.safe_angle) / 10 
                if((self.find_closest_object_angle(self.recent_scan, -math.pi/2 - math.pi /2, -math.pi/2 + math.pi/2)[1] - self.desired_distance) > .15):
                    lin = 0
                else:
                    lin = .2
                if (abs(self.find_closest_object_angle(self.recent_scan, -math.pi/2 - math.pi /2, -math.pi/2 + math.pi/2)[1] - self.desired_distance) > .05):
                    ang -= 10 * (self.find_closest_object_angle(self.recent_scan, -math.pi/2 - math.pi /2, -math.pi/2 + math.pi/2)[1] - self.desired_distance)
                    if((self.find_closest_object_angle(self.recent_scan, -math.pi/2 - math.pi /2, -math.pi/2 + math.pi/2)[1] - self.desired_distance) < 0):
                        ang -= .05
                    else:
                        ang += .05
                
            self.publish_velocity(lin,ang)
        else:
            self.get_logger().error("NO CLOSEST OBJECT FORWARD (still None)")
                
        if(self.robot_state == WALL_FOLLOW):
            pass


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
