# Super Slam Dunk

## Project Description

Describe the goal of your project, why it's interesting, what you were able to make your robot do, and what the main components of your project are and how they fit together - please include diagrams and gifs when appropriate

TODO
Autonomous maze cleanup robot with particle filter localization,  A* path planning, and color-based object detection.

## Videos

TODO

## System Architecture

Describe in detail the robotics algorithm you implemented and each major component of your project, highlight what pieces of code contribute to these main components

TODO

## ROS Node Diagram
Please include a visual diagram representing all of the ROS nodes, ROS topics, and publisher/subscriber connections present in your final project.

## Topics

| Topic | Description |
|-------|-------------|
| `/map` | Occupancy grid (subscribed) |
| `/tbXX/scan` | LiDAR data (subscribed) |
| `/particle_cloud` | Particle poses (published) |
| `/estimated_robot_pose` | Robot pose estimate (published) |
TODO: Way more ROS nodes!

## Execution
Describe how to run your code, e.g., step-by-step instructions on what commands to run in each terminal window to execute your project code.

TODO

### Build

```bash
cd ~/intro_robo_ws
colcon build --packages-select maze_cleanup
source install/setup.bash
```

### Particle Filter + RViz Visualization
```bash
ros2 launch maze_cleanup visualize_particles_launch.py
```
Opens RViz with map, particle cloud (red arrows), and estimated pose (blue arrow).

### Particle Filter Only
```bash
ros2 launch maze_cleanup particle_filter_launch.py namespace:=/tbXX
```

### Main State Machine
```bash
ros2 run maze_cleanup maze-cleanup
```

TODO: Arm stuff!

## Challenges
TODO

## Future Work
TODO

## Takeaways
TODO
