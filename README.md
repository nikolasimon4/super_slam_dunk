# Super Slam Dunk

Autonomous maze cleanup robot with particle filter localization,  A* path planning, and color-based object detection.

## Build

```bash
cd ~/intro_robo_ws
colcon build --packages-select maze_cleanup
source install/setup.bash
```

## Run

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
## Topics

| Topic | Description |
|-------|-------------|
| `/map` | Occupancy grid (subscribed) |
| `/tbXX/scan` | LiDAR data (subscribed) |
| `/particle_cloud` | Particle poses (published) |
| `/estimated_robot_pose` | Robot pose estimate (published) |

