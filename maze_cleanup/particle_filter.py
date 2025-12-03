import math
import os
import time

import numpy as np
import rclpy
from rclpy.time import Duration
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformBroadcaster, TransformListener

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

import transforms3d.euler as euler
import transforms3d.quaternions as quat


def compute_prob_zero_centered_gaussian(dist, sd):
    """Compute probability using zero-centered Gaussian distribution."""
    c = 1.0 / (sd * math.sqrt(2 * math.pi))
    return c * math.exp((-math.pow(dist, 2)) / (2 * math.pow(sd, 2)))


def get_yaw_from_pose(p):
    """A helper function that takes in a Pose object (geometry_msgs) and returns yaw."""
    q = [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
    yaw = euler.quat2euler(q, axes="sxyz")[2]
    return yaw


def wrap_angle(angle):
    """Wrap angle to the range (-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def create_pose_from_xyt(x, y, theta):
    """Create a Pose object from x, y, and theta (yaw angle)."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = 0.0

    # Convert theta to quaternion
    quaternion = quat.axangle2quat([0, 0, 1], theta)
    pose.orientation.w = quaternion[0]
    pose.orientation.x = quaternion[1]
    pose.orientation.y = quaternion[2]
    pose.orientation.z = quaternion[3]

    return pose


def transform_point_robot_to_world(robot_x, robot_y, robot_theta, local_x, local_y):
    """
    Transform a point from robot's local frame to world frame.
    """
    cos_theta = math.cos(robot_theta)
    sin_theta = math.sin(robot_theta)
    world_x = robot_x + cos_theta * local_x - sin_theta * local_y
    world_y = robot_y + sin_theta * local_x + cos_theta * local_y
    return world_x, world_y


def transform_vector_world_to_robot(world_dx, world_dy, robot_theta):
    """
    Transform a vector from world frame to robot's local frame.
    """
    cos_theta = np.cos(robot_theta)
    sin_theta = np.sin(robot_theta)
    robot_dx = cos_theta * world_dx + sin_theta * world_dy
    robot_dy = -sin_theta * world_dx + cos_theta * world_dy
    return robot_dx, robot_dy


class Particle:
    def __init__(self, pose, w):
        # Particle pose (Pose object from geometry_msgs)
        self.pose = pose
        # Particle weight
        self.w = w


class ParticleFilter(Node):
    def __init__(self):
        # Once everything is setup (including getting the map data in get_map())
        # initialized will be set to true
        self.initialized = False

        super().__init__("turtlebot4_particle_filter")

        # get the ROS_DOMAIN_ID aka robot number
        ros_domain_id = os.getenv("ROS_DOMAIN_ID")
        if int(ros_domain_id) < 10:
            ros_domain_id = "0" + ros_domain_id
        self.get_logger().info(f"ROS_DOMAIN_ID: {ros_domain_id}")

        # Set the topic names and frame names
        self.base_frame = "base_footprint"
        self.map_topic = "/map"
        self.odom_frame = "odom"
        self.scan_topic = f"/tb{ros_domain_id}/scan"

        # Initalize our map
        self.map = OccupancyGrid()
        # The number of particles used in the particle filter
        self.num_particles = 500
        # Initialize the particle cloud array
        self.particle_cloud = []
        # Initialize the estimated robot pose
        self.robot_estimate = Pose()
        # Set threshold values for linear and angular movement before we preform an update
        self.lin_mvmt_threshold = 0.05
        self.ang_mvmt_threshold = np.pi / 24
        self.odom_pose_last_motion_update = None

        # Motion model noise parameters
        self.motion_noise_linear = 0.02  # Standard deviation for linear motion noise
        self.motion_noise_angular = 0.05  # Standard deviation for angular motion noise

        # Measurement model parameters
        # Standard deviation (m) for range measurement model
        self.measurement_sd = 0.5
        self.measurement_beam_stride = 30  # 1080 / 30 = 36 beams
        self.occupancy_threshold = 50  # Cells with value >= threshold are obstacles
        self.treat_unknown_as_obstacle = False

        # Weight exponent to flatten overly confident weight distributions
        self.weight_exponent = 0.3

        # Add small random noise when resampling to keep particles spread out
        self.resample_pos_noise = 0.02
        self.resample_yaw_noise = 0.04

        # Setup publishers and subscribers
        self.particles_pub = self.create_publisher(PoseArray, f"/tb{ros_domain_id}/particle_cloud", 10)
        self.robot_estimate_pub = self.create_publisher(
            PoseStamped, f"/tb{ros_domain_id}/estimated_robot_pose", 10
        )
        qos_profile = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid, self.map_topic, self.get_map, qos_profile
        )
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.robot_scan_received,
            qos_profile_sensor_data,
        )

        # Initialize the transform buffer
        self.tf_buffer = tf2_ros.Buffer()
        # Initialize the transform listener and provide the buffer and node
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create a timer to continuously publish particle cloud for visualization
        self.publish_timer = self.create_timer(1.0, self.publish_particle_cloud_timer)

        self.get_logger().info("Particle Filter Node Initialized")

    def get_map(self, data):
        if not self.initialized:
            self.map = data
            self.get_logger().info("Map received!")
            self.initialized = True
            self.initialize_particle_cloud()

    def initialize_particle_cloud(self):
        """
        Initialize particle cloud by sampling particles uniformly in free space.
        Convert world (x, y) to grid indices with map.info.resolution and origin;
        accept only cells with occupancy values indicating free space (< 50);
        assign theta uniformly in (-pi, pi] and set initial weights to 1/N.
        """
        self.particle_cloud = []

        # Get map parameters
        map_width = self.map.info.width
        map_height = self.map.info.height
        resolution = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        # Find all free space cells (occupancy < 50)
        free_cells = []
        for i in range(map_height):
            for j in range(map_width):
                # Convert grid indices to map data index (row-major ordering)
                map_index = i * map_width + j
                if (
                    map_index < len(self.map.data)
                    and self.map.data[map_index] >= 0
                    and self.map.data[map_index] < 50
                ):
                    # Convert grid indices to world coordinates
                    # +0.5 to center in cell
                    world_x = origin_x + (j + 0.5) * resolution
                    world_y = origin_y + (i + 0.5) * resolution
                    free_cells.append((world_x, world_y))

        self.get_logger().info(
            f"Found {len(free_cells)} free cells for particle initialization"
        )

        for _ in range(self.num_particles):
            # Randomly select a free cell
            random_cell = free_cells[np.random.randint(0, len(free_cells))]

            # Assign theta uniformly in (-pi, pi]
            theta = np.random.uniform(-np.pi, np.pi)

            # Create pose for this particle
            pose = create_pose_from_xyt(random_cell[0], random_cell[1], theta)

            # Create particle with initial weight 1/N
            initial_weight = 1.0 / self.num_particles
            particle = Particle(pose, initial_weight)
            self.particle_cloud.append(particle)

        self.get_logger().info(f"Initialized {len(self.particle_cloud)} particles")

        self.publish_particle_cloud()

    def normalize_particles(self):
        """
        Normalize particle weights so they sum to 1.0.
        """
        if not self.particle_cloud:
            return

        sum_w = 0.0
        for particle in self.particle_cloud:
            sum_w += particle.w

        # Check for zero sum
        if sum_w == 0.0:
            self.get_logger().warn("Weight sum is zero, resetting to uniform weights")
            # Reset to uniform weights
            uniform_weight = 1.0 / len(self.particle_cloud)
            for particle in self.particle_cloud:
                particle.w = uniform_weight
        else:
            # Normalize weights
            for particle in self.particle_cloud:
                particle.w = particle.w / sum_w

    def publish_particle_cloud(self):
        """
        Publish the particle cloud to the particle_cloud topic.
        """
        particle_cloud_pose_array = PoseArray()
        particle_cloud_pose_array.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id="map"
        )

        # Initialize the poses list
        particle_cloud_pose_array.poses = []

        # Add each particle's pose to the poses list
        for p in self.particle_cloud:
            particle_cloud_pose_array.poses.append(p.pose)

        self.particles_pub.publish(particle_cloud_pose_array)
        self.get_logger().info(
            f"Published {len(particle_cloud_pose_array.poses)} particles to particle_cloud topic"
        )

    def publish_particle_cloud_timer(self):
        """Timer callback to continuously publish particle cloud for RViz visualization"""
        if self.particle_cloud:
            particle_cloud_pose_array = PoseArray()
            particle_cloud_pose_array.header = Header(
                stamp=self.get_clock().now().to_msg(), frame_id="map"
            )
            particle_cloud_pose_array.poses = [p.pose for p in self.particle_cloud]
            self.particles_pub.publish(particle_cloud_pose_array)

    def publish_estimated_robot_pose(self):
        robot_pose_estimate_stamped = PoseStamped()
        robot_pose_estimate_stamped.pose = self.robot_estimate
        robot_pose_estimate_stamped.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id=self.map_topic
        )
        self.robot_estimate_pub.publish(robot_pose_estimate_stamped)

    def resample_particles(self):
        """
        Resample particles with replacement according to their normalized weights.
        Creates deep copies of selected particles and applies small Gaussian jitter to position and orientation.
        Wraps orientation angles to (-pi, pi] range and converts back to quaternions.
        """
        if not self.particle_cloud:
            return

        n = len(self.particle_cloud)
        weights = np.array(
            [max(0.0, float(p.w)) for p in self.particle_cloud], dtype=float
        )
        sum_w = float(np.sum(weights))
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            weights = np.ones(n, dtype=float) / float(n)
        else:
            weights = weights / sum_w

        # Sample new particles based on their weights
        indices = np.random.choice(np.arange(n), size=n, replace=True, p=weights)

        new_particles = []
        for idx in indices:
            src = self.particle_cloud[int(idx)]

            # Get source position and orientation
            src_x = float(src.pose.position.x)
            src_y = float(src.pose.position.y)
            theta = get_yaw_from_pose(src.pose)

            # Apply small jitter
            jx = np.random.normal(0.0, float(self.resample_pos_noise))
            jy = np.random.normal(0.0, float(self.resample_pos_noise))
            jth = np.random.normal(0.0, float(self.resample_yaw_noise))

            # Create new pose with jitter
            new_x = src_x + jx
            new_y = src_y + jy
            new_theta = wrap_angle(theta + jth)
            new_pose = create_pose_from_xyt(new_x, new_y, new_theta)

            new_particles.append(Particle(new_pose, 1.0 / float(n)))

        self.particle_cloud = new_particles

    def robot_scan_received(self, data: LaserScan):
        # Wait until initialization is complete
        if not (self.initialized):
            
            return

        # We need to be able to transfrom the laser frame to the base frame
        if not self.tf_buffer.can_transform(
            self.base_frame, data.header.frame_id, data.header.stamp
        ):
            return

        # Wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated)
        timeout_duration = Duration(seconds=0.5)

        # Attempt to get the transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.odom_frame,
                data.header.stamp,
                timeout=timeout_duration,
            )
        except Exception:
            return

        # Calculate the pose of the laser distance sensor
        p = PoseStamped(
            header=Header(stamp=Time(sec=0, nanosec=0), frame_id=data.header.frame_id)
        )
        self.laser_pose = self.tf_buffer.transform(p, self.base_frame)

        # Determine where the robot thinks it is based on its odometry
        p = PoseStamped(
            header=Header(stamp=data.header.stamp, frame_id=self.base_frame),
            pose=Pose(),
        )
        self.odom_pose = self.tf_buffer.transform(p, self.odom_frame)

        # We need to be able to compare the current odom pose to the prior odom pose
        # If there isn't a prior odom pose, set the odom_pose variable to the current pose
        if not self.odom_pose_last_motion_update:
            self.odom_pose_last_motion_update = self.odom_pose
            return

        if self.particle_cloud:
            # Check to see if we've moved far enough to perform an update
            curr_x = self.odom_pose.pose.position.x
            old_x = self.odom_pose_last_motion_update.pose.position.x
            curr_y = self.odom_pose.pose.position.y
            old_y = self.odom_pose_last_motion_update.pose.position.y
            curr_yaw = get_yaw_from_pose(self.odom_pose.pose)
            old_yaw = get_yaw_from_pose(self.odom_pose_last_motion_update.pose)

            # Find the movement deltas
            x_moved = curr_x - old_x
            y_moved = curr_y - old_y
            yaw_moved = curr_yaw - old_yaw

            if (
                np.abs(x_moved) > self.lin_mvmt_threshold
                or np.abs(y_moved) > self.lin_mvmt_threshold
                or np.abs(yaw_moved) > self.ang_mvmt_threshold
            ):

                # This is where the main logic of the particle filter is carried out
                self.update_particles_with_motion_model()
                self.update_particle_weights_with_measurement_model(data)
                self.normalize_particles()
                self.resample_particles()
                self.update_estimated_robot_pose()
                self.publish_particle_cloud()
                self.get_logger().info("Particle Update")
                self.publish_estimated_robot_pose()
                self.odom_pose_last_motion_update = self.odom_pose

    def update_estimated_robot_pose(self):
        """
        Compute the robot's estimated pose from the weighted particle distribution.
        Calculates weighted mean position (x, y) using particle weights and positions.
        Calculates circular mean orientation using trigonometric averaging.
        Falls back to uniform mean if weights are degenerate. Updates self.robot_estimate
        with the computed pose and converts estimated angle to quaternion representation.
        """
        if not self.particle_cloud:
            return

        sum_w = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_cos = 0.0
        weighted_sin = 0.0

        for particle in self.particle_cloud:
            w = float(particle.w)
            x = float(particle.pose.position.x)
            y = float(particle.pose.position.y)
            theta = get_yaw_from_pose(particle.pose)

            sum_w += w
            weighted_x += w * x
            weighted_y += w * y
            weighted_cos += w * math.cos(theta)
            weighted_sin += w * math.sin(theta)

        # Fallback to uniform mean if weights are invalid
        if sum_w == 0.0:
            n = float(len(self.particle_cloud))
            weighted_x = sum(p.pose.position.x for p in self.particle_cloud) / n
            weighted_y = sum(p.pose.position.y for p in self.particle_cloud) / n
            weighted_cos = (
                sum(math.cos(get_yaw_from_pose(p.pose)) for p in self.particle_cloud)
                / n
            )
            weighted_sin = (
                sum(math.sin(get_yaw_from_pose(p.pose)) for p in self.particle_cloud)
                / n
            )
            est_x = float(weighted_x)
            est_y = float(weighted_y)
            est_theta = math.atan2(weighted_sin, weighted_cos)
        else:
            est_x = weighted_x / sum_w
            est_y = weighted_y / sum_w
            est_theta = math.atan2(weighted_sin, weighted_cos)

        # Update stored estimate
        self.robot_estimate = create_pose_from_xyt(est_x, est_y, est_theta)

    def update_particle_weights_with_measurement_model(self, data):
        """
        Update particle weights using ray-casting measurement model.
        Downsamples 36 LiDAR beams to compute laser sensor pose in world
        coordinates given particle's position/orientation.
        For each beam, calculates world-frame beam direction, casts ray using _raycast_distance()
        to predict range, compares predicted vs observed range using zero-centered Gaussian,
        and accumulates log-likelihood to avoid numerical underflow.
        Finally, it tempers weights to reduce overconfidence, yielding a robust weight for each particle.
        """
        if not self.particle_cloud or not self.initialized:
            return

        if (
            self.map.info.width == 0
            or self.map.info.height == 0
            or self.map.info.resolution <= 0.0
        ):
            return

        # Determine laser pose relative to base and its yaw offset
        # self.laser_pose is set in robot_scan_received just before this call
        laser_off_x = 0.0
        laser_off_y = 0.0
        laser_off_yaw = 0.0
        if hasattr(self, "laser_pose") and self.laser_pose is not None:
            laser_off_x = float(self.laser_pose.pose.position.x)
            laser_off_y = float(self.laser_pose.pose.position.y)
            laser_off_yaw = get_yaw_from_pose(self.laser_pose.pose)

        # Beam selection (downsample for performance)
        num_beams = len(data.ranges)
        if num_beams == 0:
            return
        stride = max(1, int(self.measurement_beam_stride))
        beam_indices = range(0, num_beams, stride)

        # Precompute step size for ray marching
        step = max(0.01, 0.5 * float(self.map.info.resolution))

        # Compute log-likelihood per particle for numerical stability
        log_weights = []
        for particle in self.particle_cloud:
            px = float(particle.pose.position.x)
            py = float(particle.pose.position.y)
            ptheta = get_yaw_from_pose(particle.pose)

            # Laser origin in world given particle pose
            laser_x, laser_y = transform_point_robot_to_world(
                px, py, ptheta, laser_off_x, laser_off_y
            )

            # Compute log-likelihood for this particle
            log_w = self._compute_particle_log_likelihood(
                laser_x, laser_y, ptheta, laser_off_yaw, data, beam_indices, step
            )
            log_weights.append(log_w)

        # Convert log-weights to linear weights in a numerically stable fashion
        self._apply_log_weights_to_particles(log_weights)

    def _compute_particle_log_likelihood(
        self, laser_x, laser_y, particle_theta, laser_off_yaw, data, beam_indices, step
    ):
        """
        Compute log-likelihood for a single particle given laser scan data.
        Returning the log-likelihood for this particle.
        """
        log_w = 0.0
        used = 0

        for bi in beam_indices:
            obs_r = data.ranges[bi]
            # Skip NaN; clamp inf to range_max
            if obs_r is None or not math.isfinite(obs_r):
                obs_r = float(data.range_max)
            obs_r = max(float(data.range_min), min(float(obs_r), float(data.range_max)))

            beam_angle = float(data.angle_min) + float(bi) * float(data.angle_increment)
            world_theta = particle_theta + laser_off_yaw + beam_angle

            pred_r = self._raycast_distance(
                laser_x, laser_y, world_theta, float(data.range_max), step
            )

            diff = obs_r - pred_r
            prob = compute_prob_zero_centered_gaussian(diff, float(self.measurement_sd))
            # Guard against log(0)
            log_w += math.log(max(prob, 1e-12))
            used += 1

        # If no beams were usable, give a tiny weight
        if used == 0:
            log_w = -1e6  # Nick, remb to test this more. --paulo

        return log_w

    def _apply_log_weights_to_particles(self, log_weights):
        """
        Convert log-weights to linear weights and apply to particles.
        Uses numerical stabilization and tempering to avoid overconfidence.
        """
        max_log_w = max(log_weights) if log_weights else 0.0
        for particle, lw in zip(self.particle_cloud, log_weights):
            # Temper weights to avoid overconfidence
            tempered = math.exp(lw - max_log_w)
            particle.w = math.pow(tempered, float(self.weight_exponent))

    def _world_to_map_indices(self, x, y):
        """
        Convert world coordinates (meters) to map indices (row i, col j).
        Returns (i, j) or None if out-of-bounds.
        """
        res = float(self.map.info.resolution)
        origin_x = float(self.map.info.origin.position.x)
        origin_y = float(self.map.info.origin.position.y)
        j = int(math.floor((x - origin_x) / res))
        i = int(math.floor((y - origin_y) / res))
        if (
            i < 0
            or j < 0
            or i >= int(self.map.info.height)
            or j >= int(self.map.info.width)
        ):
            return None
        return (i, j)

    def _is_occupied(self, i, j):
        """
        Check if a map cell is occupied (>= threshold) or unknown if configured.
        Out-of-bounds should be handled before calling this.
        """
        idx = i * int(self.map.info.width) + j
        if idx < 0 or idx >= len(self.map.data):
            return True
        val = int(self.map.data[idx])
        if val < 0:
            return True if self.treat_unknown_as_obstacle else False
        return val >= int(self.occupancy_threshold)

    def _raycast_distance(self, x, y, theta, max_range, step):
        """
        March along a ray from (x, y) at angle theta until hitting an occupied
        cell or exceeding max_range. Returns the traveled distance in meters.
        """
        r = 0.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        while r <= max_range:
            wx = x + r * cos_t
            wy = y + r * sin_t
            ij = self._world_to_map_indices(wx, wy)
            if ij is None:
                # Outside map bounds: consider as hit at boundary
                # Hitting map boundary: assume no obstacle beyond => return max_range
                return max_range
            if self._is_occupied(ij[0], ij[1]):
                return r
            r += step
        return max_range

    def update_particles_with_motion_model(self):
        """
        Apply odometry deltas delta x, delta y, delta theta from robot_scan_received().
        For each particle, rotate the translational delta from robot frame into world frame
        via the particle's theta, add Gaussian noise to translation and rotation, update pose,
        and wrap theta to (-pi, pi].
        """
        if not self.particle_cloud or not self.odom_pose_last_motion_update:
            return

        # Calculate odometry deltas in world coordinates
        curr_x = self.odom_pose.pose.position.x
        old_x = self.odom_pose_last_motion_update.pose.position.x
        curr_y = self.odom_pose.pose.position.y
        old_y = self.odom_pose_last_motion_update.pose.position.y
        curr_yaw = get_yaw_from_pose(self.odom_pose.pose)
        old_yaw = get_yaw_from_pose(self.odom_pose_last_motion_update.pose)

        # World frame deltas
        world_delta_x = curr_x - old_x
        world_delta_y = curr_y - old_y
        delta_theta = wrap_angle(curr_yaw - old_yaw)

        # Convert world frame motion to robot frame motion
        robot_delta_x, robot_delta_y = transform_vector_world_to_robot(
            world_delta_x, world_delta_y, old_yaw
        )

        # Update each particle
        for particle in self.particle_cloud:
            # Get current particle position and orientation
            particle_x = particle.pose.position.x
            particle_y = particle.pose.position.y
            particle_theta = get_yaw_from_pose(particle.pose)

            # Transform robot frame motion to world frame using particle's orientation
            particle_world_dx, particle_world_dy = transform_point_robot_to_world(
                0, 0, particle_theta, robot_delta_x, robot_delta_y
            )

            # Add Gaussian noise to motion
            noisy_delta_x = particle_world_dx + np.random.normal(
                0, self.motion_noise_linear
            )
            noisy_delta_y = particle_world_dy + np.random.normal(
                0, self.motion_noise_linear
            )
            noisy_delta_theta = delta_theta + np.random.normal(
                0, self.motion_noise_angular
            )

            # Update particle pose
            new_x = particle_x + noisy_delta_x
            new_y = particle_y + noisy_delta_y
            new_theta = wrap_angle(particle_theta + noisy_delta_theta)
            particle.pose = create_pose_from_xyt(new_x, new_y, new_theta)

        self.get_logger().info(
            f"Updated {len(self.particle_cloud)} particles with motion model: "
            f"robot_dx={robot_delta_x:.3f}, robot_dy={robot_delta_y:.3f}, dtheta={delta_theta:.3f}"
        )


def main():
    rclpy.init()
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

