import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image

import cv2
import cv_bridge
import numpy as np
import os
import time
import math
import heapq
import transforms3d.euler as euler
import transforms3d.quaternions as quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from omx_cpp_interface.msg import ArmGripperPosition
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient


# Helper functions
def get_yaw_from_pose(p: Pose):
    """A helper function that takes in a Pose object (geometry_msgs) and returns yaw."""
    q = [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
    yaw = euler.quat2euler(q, axes="sxyz")[2]
    return yaw

# Picking up joint positions
PICK_JOINT1 = 0.0
PICK_JOINT2 = 0.5
PICK_JOINT3 = -0.350
PICK_JOINT4 =  0.0


# Stowed joint positions
STOW_JOINT1 = 0.0
STOW_JOINT2 = -0.7
STOW_JOINT3 = -0.350
STOW_JOINT4 =  0.0


# Values for opening/closing the gripper
GRIPPER_OPEN = .009
GRIPPER_CLOSED = -.009

# Arrays to pass into function to make pos passing easier
ARM_PICK_UP_JOINT = [PICK_JOINT1, PICK_JOINT2, PICK_JOINT3, PICK_JOINT4]
ARM_STOW_JOINT = [STOW_JOINT1, STOW_JOINT2, STOW_JOINT3, STOW_JOINT4]

# CONSTANTS
INIT                           = "INIT"
LOAD_SLAM_MAP                  = "LOAD_SLAM_MAP"
LOCALIZE_WITH_FILTER           = "LOCALIZE_WITH_FILTER"
WALL_FOLLOW                    = "WALL_FOLLOW"
OBSERVE_OBJECT                 = "OBSERVE_OBJECT"
WAIT_FOR_TARGET                = "WAIT_FOR_TARGET"
PLAN_PATH_TO_OBJECT            = "PLAN_PATH_TO_OBJECT"
NAVIGATE_TO_OBJECT             = "NAVIGATE_TO_OBJECT"
MOVE_TO_SEEN                   = "MOVE_TO_SEEN"
ALIGN_WITH_OBJECT              = "ALIGN_WITH_OBJECT"
PICK_UP_OBJECT                 = "PICK_UP_OBJECT"
PLAN_PATH_TO_BIN               = "PLAN_PATH_TO_BIN"
NAVIGATE_TO_BIN                = "NAVIGATE_TO_BIN"
DROP_OBJECT                    = "DROP_OBJECT"
TASK_COMPLETE                  = "TASK_COMPLETE"

OBJECTS = ["blue", "green", "pink"]

class ObjectCollector(Node):

    def __init__(self):
        super().__init__('maze_object_collector')

        # Pull turtlebot number and pad with leading zero if needed
        unformatted = os.getenv('ROS_DOMAIN_ID')
        ros_domain_id = f'{int(unformatted):02d}'
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')

        self.bridge = cv_bridge.CvBridge()
        
        #Wall following
        self.desired_distance = .25 # Desired dist from wall when following
        self.recent_scan = None
        self.counter = 0
        self.countdown = 1000
        # State machine
        self.objects_count = 0
        self.robot_state = INIT
        self.map_loaded = False
        self.map = None
        self.inflated_map = None # Map that has inflated walls to improve A* pathing
        self.inflation_radius = 0.2 # Tunable for how much inflation we want  
        
        
        self.desired_distance_color = .35 # Desired distance from colored object before picking up (used for approaching object)

        # Object tracking
        self.object_positions = {"blue": None, "green": None, "pink": None}
        self.objects_found    = {"blue": False, "green": False, "pink": False}
        # TO TEST A* CODE:
        """
        SET pink to true in objects_found
        Create Pose() object for pink with desired x and y coordinates and put it into self.objects_found
        Create Pose() object for robot and set self.robot_pose to the desired starting point (should be somewhere you can put the robot relatively confidently in it's pose)
        In handle_load_slam map, instead of the localize_with_filter phase, set the robot state to WAIT_FOR_TARGET state
        At this point the robot will set the object to pink (handle_wait_for_target()) and then go to the plan_path state
        This should then just run as if the robot has the object and pose correct. after it moves a bit the pose should update to something more useful and you should be able to rerun the a* code
        That said it is fine if the robot moves really slowly during A* to minimize error 
        """
        # Stores information on each detected object (found is false if obj isn't detected)
        self.detected_objects = {
            'pink': {'found': False, 'cx': 0, 'cy': 0, 'area': 0},
            'green': {'found': False, 'cx': 0, 'cy': 0, 'area': 0},
            'blue': {'found': False, 'cx': 0, 'cy': 0, 'area': 0}
        }

        # Same as above but for tags not in use
        self.detected_tags = {
            1: {'found': False, 'cx': 0, 'cy': 0, 'width': 0},
            2: {'found': False, 'cx': 0, 'cy': 0, 'width': 0},
            3: {'found': False, 'cx': 0, 'cy': 0, 'width': 0}
        }

        # Runtime state variables
        self.target_object = None
        self.current_path = []
        self.has_object = False
        self.robot_pose = Pose()
        self.robot_pose.orientation.w = 0.0
        self.robot_pose.orientation.x = 0.0
        self.robot_pose.orientation.y = 0.0
        self.robot_pose.orientation.z = 0.0
        self.robot_pose.position.x = 0.0
        self.robot_pose.position.y = 0.0
        self.safe_angle = -math.pi / 2
        self.closest_object_forward = None
        self.closest_object_left = None 
        qos_profile = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Tunable constants for image and searching
        self.image_width = 640 # changed later according to the actual received image
        self.image_height = 480 
        self.kp_angular = 1.0
        self.kp_linear = 1.0
        self.approach_area_threshold = 100 # colored objects
        self.approach_width_threshold = 100 # AR tags
        self.center_threshold = 50 # pixels

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

        # Publisher
        self.cmd_pub = self.create_publisher(
            Twist, 
            f"/tb{ros_domain_id}/cmd_vel",
            10
        )

        # State machine timer 
        self.create_timer(0.1, self.state_machine_step)

        self.get_logger().info("ObjectCollector Node Initialized")

    def pose_callback(self, pose: PoseStamped):
        """
        Callback for pose subscription
        """ 
        self.robot_pose = pose.pose
    # Map loading
    def get_map_callback(self, msg: OccupancyGrid):
        """Receive a SLAM map from /map and store it."""
        if not self.map_loaded:
            self.map = msg
            self.map_loaded = True
            self.get_logger().info("SLAM map received from /map.")

    # Scan callback
    def scan_callback(self, msg):
        self.closest_object_forward = self.find_closest_object_angle(msg,-math.pi/8,math.pi/8)
        self.closest_object_left = self.find_closest_object_angle(msg,-3 * math.pi / 4, -1 * math.pi/4)
        self.recent_scan = msg
        
        
        
        if(self.robot_state == LOCALIZE_WITH_FILTER or self.robot_state == WALL_FOLLOW):
            if(self.objects_count == 3 and self.countdown <= 0):
                self.robot_state = WAIT_FOR_TARGET
                return 
            self.wall_follow_publish()
        if(self.robot_state == OBSERVE_OBJECT and self.target_object and not self.detected_objects[self.target_object]['found']):
            self.get_logger().info(f"GETTING OBJECT {self.target_object}")
            f_width = math.pi/20 
            
            while(True):
                ang, dist = self.find_closest_object_angle(msg,-f_width,f_width)
                if(abs(abs(ang) - f_width) < f_width / 2):
                    self.get_logger().info("Object detected, but currently detecting wall with lidar scan, decreasing forward looking distance")
                    f_width = f_width / 2
                else:
                    if (dist>= 10):
                        self.objects_found[self.target_object] = False
                        self.target_object = None
                        self.robot_state = WALL_FOLLOW
                        self.get_logger().info("Object not aligned, continuing to look")
                        return
                    ob_pose = Pose()
                    ob_pose.orientation = self.robot_pose.orientation
                    ob_pose.position = self.robot_pose.position
                    self.object_positions[self.target_object] = ob_pose
                    self.get_logger().info(f"GOT OBJECT {self.target_object} POSITION {ob_pose.position.x},{ob_pose.position.y} DISTANCE {dist}")
                    self.objects_count += 1
                    self.target_object = None
                    self.robot_state = WALL_FOLLOW
                    return
        forward_obj_ang, forward_obj_dist = self.find_closest_object_angle(msg,-math.pi/20,math.pi/20)

        # If the robot is approaching and within the desired distance stops the robot and updates the state
        if (self.robot_state == MOVE_TO_SEEN and forward_obj_dist < self.desired_distance_color):
            
            self.robot_state = PICK_UP_OBJECT
            self.publish_velocity(0.0,0.0)
            return         
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
        if(self.counter > 5):
            localization_done = True
        else:
            self.counter += 1
            self.publish_velocity(.1,0) 
            localization_done = False
        if localization_done:
            self.robot_state = WALL_FOLLOW

    def handle_wall_follow(self):
        if(self.objects_count == 3):
            self.countdown -= 1
        pass

    def handle_observe_object(self):
        self.get_logger().info("Observing object...")
        pass

    def handle_wait_for_target(self):
        self.target_object = "pink"
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
        pass
    def handle_move_to_seen_object(self):
        # Finds position of object/tag relative to robot in image
        if self.target_object is not None and self.detected_objects[self.target_object]['found']:
            cx = self.detected_objects[self.target_object]['cx'] - self.image_width / 2
        else:
            cx = 0
            self.get_logger().info(f"Object/tag not found, target obj {self.target_object}, target tag: {self.target_tag}")
        # Moves robot forward
        self.publish_velocity(.1 , 0)
        time.sleep(.1)
        # Attempts to move slowly to keep robot aligned to object
        self.publish_velocity(0, -(cx / 50))

    def handle_align_with_object(self):
        refined = True
        if refined:
            self.robot_state = PICK_UP_OBJECT

    def handle_pick_up_object(self):
        # Closes gripper
        self.send_gripper_command(GRIPPER_CLOSED)
        # Stows gripper
        self.publish_joint_angles(ARM_STOW_JOINT)
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

    # Helper functions
 
    def publish_velocity(self, lin, ang):
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)

    def build_inflated_map(self):
        """
        Build an A* inflated map to make sure pathing stays away from walls.
        """
        if self.map is None:
            self.get_logger().error("No map available to build inflated map.")
            return

        width = self.map.info.width
        height = self.map.info.height
        res = self.map.info.resolution
        data = self.map.data

        inflated = [0] * (width * height)

        # Copy map obstacles
        for i, v in enumerate(data):
            if v != 0:
                inflated[i] = 100

        # Determine inflation radius in cells
        radius_cells = int(self.inflation_radius / res)
        if radius_cells <= 0:
            self.inflated_map = inflated
            return

        # Inflate obstacles
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if inflated[idx] == 100:
                    for dy in range(-radius_cells, radius_cells + 1):
                        for dx in range(-radius_cells, radius_cells + 1):
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                nidx = ny * width + nx
                                inflated[nidx] = 100

        self.inflated_map = inflated
        self.get_logger().info("Inflated A* map built")

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

        # Make sure inflated map is built
        if self.inflated_map is None:
            self.build_inflated_map()

        # Bounds check
        if not (0 <= start_idx < width * height) or not (0 <= goal_idx < width * height):
            self.get_logger().error("Start or goal index out of bounds.")
            return []

        # Collision check
        if self.inflated_map[start_idx] == 100:
            self.get_logger().error("Start index is an obstacle.")
            return []
        if self.inflated_map[goal_idx] == 100:
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
                if self.inflated_map[nb] == 100:
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

                    if (self.inflated_map[mid1_idx] == 100 or
                        self.inflated_map[mid2_idx] == 100):
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
    
    
    ### GRIPPER HELPERS
    def send_gripper_command(self, position, effort=1.0):
        """
        sends command to gripper to move to given position with given effort

        Args:
            position (int): Position of gripper (see constants above)
            effort (float, optional): Torque value, default 1 is fine. Defaults to 1.0.
        """
        goal = GripperCommand.Goal()
        goal.command.position = position 
        goal.command.max_effort = effort 
        self.get_logger().info(f'Sending gripper command: position={position}')
        self.gripper_pub.send_goal_async(goal)
        time.sleep(2)        

    def publish_joint_angles(self, joint_angles, duration=1.0):
        """Publishes given joint angles to arm

        Args:
            joint_angles (list of joint angles): list of joint angles to be published in order (j1,j2...)
            duration (float, optional): Duration of movement, default is good enough. Defaults to 1.0.
        """
        
        # If the joint angle input doesn't match to expected names, prints an error
        if len(joint_angles) != len(self.joint_names):
            self.get_logger().error(f"Expected {len(self.joint_names)} joint angles, got {len(joint_angles)}")
            return

        # Constructing joint_trajectory message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1.0) * 1e9)

        msg.points.append(point)

        self.joint_publisher.publish(msg)
        self.get_logger().info(f"Sent joint trajectory: {joint_angles}")
        time.sleep(2)

    
    
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
            # If the wall is in front, turn until forward is clear
            if(self.closest_object_forward[1] < self.desired_distance + .05):
                ang = 1.5
                lin = 0
            else:
                # Set turning based on closest object on right side of robot
                ang = (self.find_closest_object_angle(self.recent_scan, -math.pi/2 - math.pi /2 + math.pi/50, -math.pi/2 + math.pi/2 - math.pi/50)[0] - self.safe_angle) / 10 
                # If the object is far away from a wall, turn until there is a wall nearby (eg 270 deg corner)
                if((self.find_closest_object_angle(self.recent_scan, -math.pi/2 - math.pi /2, -math.pi/2 + math.pi/2)[1] - self.desired_distance) > .15):
                    lin = 0
                else:
                    lin = .2
                # If the distance error is too far, add to the angular movement
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
            

    def image_callback(self, msg):
        """
        Deals with image every time one is recieved from camera
        """

        # converts the incoming ROS message to OpenCV format and HSV
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image_height, self.image_width = image.shape[:2]
        image = image[int(self.image_height * 0.75):self.image_height, 0:self.image_width]
        self.image_height = self.image_height * .25
        # detect colored tubes
        self.detect_colored_tubes(image, hsv)
        for color, info in self.detected_objects.items():
            if info['found']:    
                if not self.objects_found[color]:
                    self.get_logger().info(f"color: {color}, cx: {info['cx']}, cy: {info['cy']}, area: {info['area']}")
                    self.target_object = color
                    self.objects_found[color] = True
                    self.robot_state = OBSERVE_OBJECT
                    self.publish_velocity(0,0)
                    break
                
        # detect aruco tags (currently not in use)
        # self.detect_aruco_tags(image, gray)

        
        # show debug image
        debug_image = self.draw_debug_info(image)
        cv2.imshow("debug_window", debug_image)
        cv2.waitKey(1)
        

                
        # If the robot is looking for an object and has found the object, aligns the robot
        if  self.target_object is not None and self.robot_state == OBSERVE_OBJECT:
            # Gets the position of object in image
            c_x_obj = self.detected_objects[self.target_object]['cx'] - self.image_width / 2 
            self.get_logger().info(f"Aligning to tag {self.target_object}, angle distance from tag {c_x_obj}")
            # If robot is aligned, switches to approaching state and updates arm position to be able to pick up obj
            if abs(c_x_obj) < 6:
                self.publish_velocity(0.0,0.0)
                self.detected_objects = {
                    'pink': {'found': False, 'cx': 0, 'cy': 0, 'area': 0},
                    'green': {'found': False, 'cx': 0, 'cy': 0, 'area': 0},
                    'blue': {'found': False, 'cx': 0, 'cy': 0, 'area': 0}
                }
                return 
            if self.detected_objects[self.target_object]['found']:
                # Spins the robot slightly to center it
                self.publish_velocity(0,-c_x_obj/200)  
            else:
                self.get_logger().info(f"DOESNT SEE obj")
                
        if  self.target_object is not None and self.robot_state == NAVIGATE_TO_OBJECT and self.detected_objects[self.target_object]['found']:
            self.robot_state = ALIGN_WITH_OBJECT
            self.publish_velocity(0.0,0.0)
            # Gets the position of object in image
            c_x_obj = self.detected_objects[self.target_object]['cx'] - self.image_width / 2 
            self.get_logger().info(f"Aligning to tag {self.target_object}, angle distance from tag {c_x_obj}")
            # If robot is aligned, switches to approaching state and updates arm position to be able to pick up obj
            if abs(c_x_obj) < 6:
                self.publish_velocity(0.0,0.0)
                self.robot_state = MOVE_TO_SEEN
                self.send_gripper_command(GRIPPER_OPEN)
                self.publish_joint_angles(ARM_PICK_UP_JOINT)

                return 
            if self.detected_objects[self.target_object]['found']:
                # Spins the robot slightly to center it
                self.publish_velocity(0,-c_x_obj/200)  
            else:
                self.get_logger().info(f"DOESNT SEE obj")


            

    def detect_colored_tubes(self, image, hsv):

        color_ranges = {'pink': (np.array([140, 50, 200]), np.array([160, 255, 255])),
                        'green': (np.array([35, 100, 100]), np.array([45, 255, 255])),
                        'blue': (np.array([90, 80, 80]), np.array([95, 210, 210]))}
        
        kernel = np.ones((5, 5), np.uint8)

        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            M = cv2.moments(mask)

            if M['m00'] > 5000:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                area = M['m00']

                self.detected_objects[color] = {
                    'found': True,
                    'cx': cx,
                    'cy': cy,
                    'area': area
                }

                cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(image, f'{color}: {area:.0f}', (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                self.detected_objects[color]['found'] = False


    def detect_aruco_tags(self, image, gray):
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_parameters)

        for tag_id in [1, 2, 3]:
            self.detected_tags[tag_id]['found'] = False
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

            for i, tag_id in enumerate(ids.flatten()):
                if tag_id in [1, 2, 3]:
                    corner_points = corners[i][0]

                    cx = int(np.mean(corner_points[:, 0]))
                    cy = int(np.mean(corner_points[:, 1]))

                    width = np.linalg.norm(corner_points[0] - corner_points[1])

                    self.detected_tags[tag_id] = {
                        'found': True,
                        'cx': cx,
                        'cy': cy,
                        'width': width
                    }

                    if self.has_object and tag_id == self.target_tag and self.initial_tag_position is None:
                        center_offset = cx - (self.image_width // 2)
                        if center_offset < -20:
                            self.initial_tag_position = 'left'
                            self.get_logger().info(f'Tag {tag_id} initially detected on left')
                        elif center_offset > 20:
                            self.initial_tag_position = 'right'
                            self.get_logger().info(f'Tag {tag_id} initially detected on right')
                        else:
                            self.initial_tag_position = 'center'
                            self.get_logger().info(f'Tag {tag_id} initially detected on center')

                    cv2.putText(image, f'Tag {tag_id}: {width:.0f}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    
    def draw_debug_info(self, image):
        h, w = image.shape[:2]
        cv2.line(image, (w//2, 0), (w//2, h), (255, 0, 0), 2)

        cv2.putText(image, f'State: {self.robot_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f'State: Arm_Obj: {self.has_object}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.target_object:
            cv2.putText(image, f'Target: {self.target_object} -> Tag None', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image
    

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
