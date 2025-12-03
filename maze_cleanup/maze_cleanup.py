import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
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
        # State machine
        self.robot_state = INIT
        self.map_loaded = False
        self.map = None
        self.inflated_map = None # Map that has inflated walls to improve A* pathing
        self.inflation_radius = 0.2 # Tunable for how much inflation we want  

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
        self.robot_pose = pose
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
        if(self.robot_state == LOCALIZE_WITH_FILTER or self.robot_state == WALL_FOLLOW):
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
