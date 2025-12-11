# Super Slam Dunk
Names

## Project Description

Describe the goal of your project, why it's interesting, what you were able to make your robot do, and what the main components of your project are and how they fit together - please include diagrams and gifs when appropriate

TODO
Autonomous maze cleanup robot with particle filter localization,  A* path planning, and color-based object detection.

### Videos

TODO

## System Architecture

### Overview

The Super Slam Dunk system is an end-to-end autonomous robotics pipeline implemented  
as two separate ROS 2 nodes: one node for localization using a particle filter and  
one node implementing the main state machine (`ObjectCollector`).

The robot operates in a maze environment using a TurtleBot equipped with an RGB  
camera, LiDAR, and an OpenManipulator arm.

All robot behaviors are coordinated through a **state machine**, which integrates  
perception, mapping, planning, navigation, and manipulation into one unified  
control structure.

The robot is capable of exploring the environment, detecting colored objects,  
planning collision-free paths using A*, and picking up and placing the objects in  
specified locations.

---

### High-Level Control and State Machine

The core of the system is a state machine implemented in the `ObjectCollector`  
node.

The main execution loop of the state machine is handled by:

- `state_machine_step()`

This state machine governs all high-level robot behavior, including  
initialization, exploration, object detection, path planning, navigation, and  
manipulation.

Each state corresponds to a specific robot behavior, and transitions occur based  
on sensor feedback, task completion, or failure conditions.

Key states include initialization and map loading, wall-following exploration,  
object observation and alignment, path planning to objects and bins, autonomous  
navigation along planned paths, object pickup, object drop-off, and task  
completion. These behaviors are implemented in state handler functions such as:

- `handle_init()`  
- `handle_load_slam_map()`  
- `handle_wall_follow()`  
- `handle_observe_object()`  
- `handle_plan_to_object()`  
- `handle_navigate_to_object()`  
- `handle_pick_up_object()`  
- `handle_plan_to_bin()`  
- `handle_navigate_to_bin()`  
- `handle_drop_object()`  

The state machine executes periodically using a ROS timer created with  
`create_timer()`. In addition to the timer-driven logic, sensor callbacks such as  
`scan_callback()` and `image_callback()` also perform state-dependent actions,  
including alignment, distance checks, and state transitions.

This design enables robust task sequencing and allows the robot to recover by  
returning to exploration or replanning if an error occurs.

---

### Mapping and Localization Integration

The system relies on an externally generated SLAM map published as an occupancy  
grid and a continuously updated estimated robot pose provided by the particle  
filter node.

Map data is received and stored using:

- `get_map_callback()`

Robot pose updates are handled by:

- `pose_callback()`

The occupancy grid provides a global representation of the maze environment, while  
the pose estimate gives the robot’s position in the map frame.

Localization accuracy is improved implicitly through wall-following exploration,  
which provides consistent motion and sensor observations that help the underlying  
particle filter converge without requiring a dedicated localization phase in the  
state machine.

---

### Perception System

#### Vision-Based Object Detection

Object perception is handled using RGB camera data processed with OpenCV.

Camera images are received using:

- `image_callback()`

The system detects colored poles (pink, green, and blue) using HSV color  
thresholding and morphological filtering implemented in:

- `detect_colored_tubes()`

For each detected object, the image centroid and contour area are computed and  
stored in a shared dictionary (`self.detected_objects`).

Object detection runs continuously during exploration and navigation. When an  
object is detected for the first time, its color and image-space position are  
recorded and used to trigger alignment and pose estimation behaviors within the  
state machine.

For detection, only the lower half of the camera image is used to prevent the  
robot from detecting objects outside the bounds of the maze.

#### LiDAR-Based Distance Sensing

LiDAR data is used for obstacle detection, wall following, and estimating the  
distance to objects during approach and drop-off.

Laser scan data is processed in:

- `scan_callback()`

Distance and angle information is extracted using:

- `find_closest_object_angle()`

These distance measurements are used to maintain safe motion, guide wall  
following, and determine when the robot is close enough to pick up or drop an  
object.

---

### Exploration and Wall Following

Exploration is achieved using a wall-following controller that allows the robot  
to traverse the maze safely while collecting perception data.

Wall-following behavior is implemented in:

- `wall_follow_publish()`

The controller maintains a desired distance from nearby walls using LiDAR  
feedback and applies proportional corrections to linear and angular velocity  
commands via:

- `publish_velocity()`

This behavior ensures coverage of the maze, improves localization quality, avoids  
obstacles, and enables the robot to discover objects without requiring prior  
knowledge of their locations.

Wall following remains active until required objects are detected or a navigation  
task is initiated.

---

### Object Localization and Cataloging

When the robot observes a colored object, it visually aligns itself with the  
object using camera feedback processed in `image_callback()`.

Once aligned, the robot records the object’s position in the map frame using its  
current pose estimate (`self.robot_pose`).

Each object’s color and map-frame position are stored in an object dictionary  
(`self.object_positions`), which is later used for path planning and task  
execution.

Detected object positions are visualized in RViz using:

- `publish_object_marker()`

This allows verification of object localization accuracy during testing.

---

### Path Planning (A*)

Global path planning is performed using a custom implementation of the A*  
algorithm operating on a grid-based representation of the occupancy map.

Map preprocessing and obstacle inflation are handled by:

- `build_obstacle_map()`

A* search over the grid is implemented in:

- `astar_indices()`

Paths are reconstructed using:

- `reconstruct_indices_path()`

The main planning interface used by the state machine is:

- `plan_path()`

The planner supports eight-connected grid motion, diagonal movement with  
corner-cutting prevention, and turn penalties to encourage smoother paths.

Planning is triggered when navigating to a detected object or transporting an  
object to a bin.

The resulting grid path is converted back into world-frame waypoints for  
execution.

---

### Path Execution and Navigation Control

Planned paths are executed using a waypoint-following navigation controller  
implemented in:

- `handle_follow_path()`

The robot drives toward each waypoint using proportional angular control and a  
constant forward velocity, publishing commands via `publish_velocity()`.

Waypoint and goal tolerances determine when the robot advances to the next  
waypoint or completes the path.

Upon reaching the final goal, the state machine transitions to the appropriate  
next state, such as object pickup or drop-off.

---

### Manipulation System

Object manipulation is handled using the OpenManipulator arm and gripper.

Arm motion is commanded by publishing joint trajectories using:

- `publish_joint_angles()`

The gripper is controlled using a ROS action client implemented in:

- `send_gripper_command()`

Predefined joint configurations are used for general movement, object pickup, and  
arm stowing.

Manipulation behaviors are tightly integrated with perception and navigation.  
Visual alignment ensures the robot is centered on the object, LiDAR distance  
thresholds determine when to initiate pickup or drop-off, and the arm is safely  
stowed before navigation resumes.

---

### Visualization and Debugging Tools

The system publishes multiple visualization markers to RViz, including planned  
paths, goal locations, and detected object positions, using:

- `publish_path_visualization()`  
- `publish_goal_marker()`  
- `clear_path_visualization()`  

These visualizations provide insight into the robot’s internal planning and  
perception processes.

An OpenCV debug window displays live camera images with detection overlays for  
color detection tuning, alignment guides, and current state information. This  
debug view is generated using:

- `draw_debug_info()`

This supports real-time debugging and parameter tuning.

---

### Summary

The Super Slam Dunk system architecture integrates perception, mapping, planning,  
navigation, and manipulation into a unified autonomous pipeline controlled by a  
state machine.


## ROS Node Diagram

### Node Diagram:

![IMG_1298](https://github.com/user-attachments/assets/169f9d23-5f80-4e24-8a4b-a3854cb69d7c)


## Tables for reference 

| Node | Description |
|------|-------------|
| **Robot Node** | Provides raw sensing and motion interfaces, including LiDAR, camera, TF transforms, and mobile base velocity execution |
| **Particle Filter Node** | Performs localization using LiDAR, odometry, and the occupancy grid map to estimate the robot pose |
| **Object Collector Node** | Main state machine node handling perception, exploration, A* path planning, navigation, and manipulation coordination |
| **Map Node** | Provides the global occupancy grid map and visualization tools (SLAM + RViz) |
| **Arm Node** | Controls the OpenManipulator arm and gripper |

---

## Robot Node

### Published Topics

| Topic | Description |
|------|-------------|
| `/tbXX/scan` | LiDAR scan data |
| `/tf` | Coordinate frame transforms |
| `/tbXX/oakd/rgb/preview/image_raw/compressed` | RGB camera images |

### Subscribed Topics

| Topic | Description |
|------|-------------|
| `/tbXX/cmd_vel` | Velocity commands for the mobile base |

---

## Particle Filter Node

### Subscribed Topics

| Topic | Description |
|------|-------------|
| `/map` | Occupancy grid map from the Map Node |
| `/tbXX/scan` | LiDAR scan data |
| `/tf` | Coordinate frame transforms |
| `/odom` | Odometry data for particle filter code |
### Published Topics

| Topic | Description |
|------|-------------|
| `/tbXX/particle_cloud` | Particle poses for localization visualization |
| `/tbXX/estimated_robot_pose` | Estimated robot pose in the map frame |

---

## Object Collector Node

### Subscribed Topics

| Topic | Description |
|------|-------------|
| `/map` | Occupancy grid map |
| `/tbXX/scan` | LiDAR scan data |
| `/tbXX/oakd/rgb/preview/image_raw/compressed` | RGB camera images |
| `/tbXX/estimated_robot_pose` | Robot pose estimate from particle filter |
| `/clicked_point` | RViz click input for testing |

### Published Topics

| Topic | Description |
|------|-------------|
| `/tbXX/cmd_vel` | Velocity commands for navigation |
| `/planned_path` | RViz visualization of planned A* path |
| `/astar_path` | Planned navigation path |
| `/goal_marker` | Navigation goal marker |
| `/detected_objects` | Detected object markers |

---

## Arm Node

### Subscribers

| Action | Description |
|--------|-------------|
| `/gripper_controller/gripper_cmd` | Open/close control of the gripper |
| `/arm_controller/joint_trajectory` | Joint trajectory commands for the arm |

---

## Map Node

### Published Topics

| Topic | Description |
|------|-------------|
| `/map` | Global occupancy grid generated by SLAM |

### Subscribed Topics

| Topic | Description |
|------|-------------|
| `/planned_path` | Path visualization |
| `/astar_path` | Planned path visualization |
| `/goal_marker` | Goal marker visualization |
| `/detected_objects` | Object marker visualization |
| `/tbXX/particle_cloud` | Particle filter visualization |
| `/estimated_robot_pose` | Estimated robot pose |

## Execution
Describe how to run your code, e.g., step-by-step instructions on what commands to run in each terminal window to execute your project code.

### Build

```bash
cd ~/intro_robo_ws
colcon build --symlink-install --packages-select maze_cleanup
source install/setup.bash
```

### Terminal 1: Launch RViz
To show map, particle filter with estimated pose, object locations, and A* paths
```bash
ros2 launch maze_cleanup visualize_particles_launch.py
```

### Terminal 2: Publish Particles
```bash
ros2 launch maze_cleanup particle_filter_launch.py namespace:=/tbXX
```

### Terminal 3: TITLE (state machine? Run Maze Cleanup?)
```bash
ros2 run maze_cleanup maze-cleanup
```

### Terminal X: OTHER TERMINALS FOR ARMS?
TODO: Arm stuff!

## Challenges
TODO

## Future Work
TODO

## Takeaways
TODO
