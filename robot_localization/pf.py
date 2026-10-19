#!/usr/bin/env python3


""" This is the starter code for the robot localization project """


import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler


class Particle(object):
   """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
       Attributes:
           x: the x-coordinate of the hypothesis relative to the map frame
           y: the y-coordinate of the hypothesis relative ot the map frame
           theta: the yaw of the hypothesis relative to the map frame
           w: the particle weight (the class does not ensure that particle weights are normalized
   """


   def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
       """ Construct a new Particle
           x: the x-coordinate of the hypothesis relative to the map frame
           y: the y-coordinate of the hypothesis relative ot the map frame
           theta: the yaw of KeyboardInterruptthe hypothesis relative to the map frame
           w: the particle weight (the class does not ensure that particle weights are normalized """
       self.w = w
       self.theta = theta
       self.x = x
       self.y = y


   def as_pose(self):
       """ A helper function to convert a particle to a geometry_msgs/Pose message """
       q = quaternion_from_euler(0, 0, self.theta)
       return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                   orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))


   # TODO: define additional helper functions if needed


class ParticleFilter(Node):
   """ The class that represents a Particle Filter ROS Node
       Attributes list:
           base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
           map_frame: the name of the map coordinate frame (should be "map" in most cases)
           odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
           scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
           n_particles: the number of particles in the filter
           d_thresh: the amount of linear movement before triggering a filter update
           a_thresh: the amount of angular movement before triggering a filter update
           pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
           particle_pub: a publisher for the particle cloud
           last_scan_timestamp: this is used to keep track of the clock when using bags
           scan_to_process: the scan that our run_loop should process next
           occupancy_field: this helper class allows you to query the map for distance to closest obstacle
           transform_helper: this helps with various transform operations (abstracting away the tf2 module)
           particle_cloud: a list of particles representing a probability distribution over robot poses
           current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                  The pose is expressed as a list [x,y,theta] (where theta is the yaw)
           thread: this thread runs your main loop
   """
   def __init__(self):
       super().__init__('pf')
       self.base_frame = "base_footprint"   # the frame of the robot base
       self.map_frame = "map"          # the name of the map coordinate frame
       self.odom_frame = "odom"        # the name of the odometry coordinate frame
       self.scan_topic = "scan"        # the topic where we will get laser scans from


       self.n_particles = 300          # the number of particles to use


       self.d_thresh = 0.2             # the amount of linear movement before performing an update
       self.a_thresh = math.pi/6       # the amount of angular movement before performing an update


       # TODO: define additional constants if needed


       # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
       self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)


       # publish the current particle cloud.  This enables viewing particles in rviz.
       self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)


       # laser_subscriber listens for data from the lidar
       self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)


       # this is used to keep track of the timestamps coming from bag files
       # knowing this information helps us set the timestamp of our map -> odom
       # transform correctly
       self.last_scan_timestamp = None
       # this is the current scan that our run_loop should process
       self.scan_to_process = None
       # your particle cloud will go here
       self.particle_cloud = []


       self.current_odom_xy_theta = []
       self.occupancy_field = OccupancyField(self)
       self.transform_helper = TFHelper(self)


       # we are using a thread to work around single threaded execution bottleneck
       thread = Thread(target=self.loop_wrapper)
       thread.start()
       self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)


   def pub_latest_transform(self):
       """ This function takes care of sending out the map to odom transform """
       if self.last_scan_timestamp is None:
           return
       postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
       self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)


   def loop_wrapper(self):
       """ This function takes care of calling the run_loop function repeatedly.
           We are using a separate thread to run the loop_wrapper to work around
           issues with single threaded executors in ROS2 """
       while True:
           self.run_loop()
           time.sleep(0.1)


   def run_loop(self):
       """ This is the main run_loop of our particle filter.  It checks to see if
           any scans are ready and to be processed and will call several helper
           functions to complete the processing.
          
           You do not need to modify this function, but it is helpful to understand it.
       """
       if self.scan_to_process is None:
           return
       msg = self.scan_to_process


       (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                          self.base_frame,
                                                                          msg.header.stamp)
       if new_pose is None:
           # we were unable to get the pose of the robot corresponding to the scan timestamp
           if delta_t is not None and delta_t < Duration(seconds=0.0):
               # we will never get this transform, since it is before our oldest one
               self.scan_to_process = None
           return
      
       (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
       print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
       # clear the current scan so that we can process the next one
       self.scan_to_process = None


       self.odom_pose = new_pose
       new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
       print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))


       if not self.current_odom_xy_theta:
           self.current_odom_xy_theta = new_odom_xy_theta
       elif not self.particle_cloud:
           # now that we have all of the necessary transforms we can update the particle cloud
           self.initialize_particle_cloud(msg.header.stamp)
       elif self.moved_far_enough_to_update(new_odom_xy_theta):
           # we have moved far enough to do an update!
           self.update_particles_with_odom()    # update based on odometry
           self.update_particles_with_laser(r, theta)   # update based on laser scan
           self.update_robot_pose()                # update robot's pose based on particles
           self.resample_particles()               # resample particles to focus on areas of high density
       # publish particles (so things like rviz can see them)
       self.publish_particles(msg.header.stamp)


   def moved_far_enough_to_update(self, new_odom_xy_theta):
       return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
              math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
              math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh




   def update_robot_pose(self):
       """ Update the estimate of the robot's pose given the updated particles.
           There are two logical methods for this:
               (1): compute the mean pose
               (2): compute the most likely pose (i.e. the mode of the distribution)
       """
       # first make sure that the particle weights are normalized
       self.normalize_particles()


       # TODO: assign the latest pose into self.robot_pose as a geometry_msgs.Pose object
       # just to get started we will fix the robot's pose to always be at the origin
       self.robot_pose = Pose()
       if hasattr(self, 'odom_pose'):
           self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                           self.odom_pose)
       else:
           self.get_logger().warn("Can't set map->odom transform since no odom data received")


   def update_particles_with_odom(self):
       """ Update the particles using the newly given odometry pose.
           The function computes the value delta which is a tuple (x,y,theta)
           that indicates the change in position and angle between the odometry
           when the particles were last updated and the current odometry.
       """
       new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
       # compute the change in x,y,theta since our last update
       if self.current_odom_xy_theta:
           old_odom_xy_theta = self.current_odom_xy_theta
           delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                    new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                    new_odom_xy_theta[2] - self.current_odom_xy_theta[2])


           self.current_odom_xy_theta = new_odom_xy_theta
       else:
           self.current_odom_xy_theta = new_odom_xy_theta
           return


       # TODO: modify particles using delta


   def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample in helper_functions.py.
        """
        # make sure the distribution is normalized
        self.normalize_particles()
        # TODO: unit test this function to make sure it is working correctly

        # Initialize the random number generator
        rng = np.random.default_rng()

        # num_rand_particles
        particle_weights = self.particle_cloud[3,:]  # Assign weights which are stored in the 4th row of the particle cloud

        # Normalize the weights to ensure they sum to 1
        particle_weights /= np.sum(particle_weights)

        # Resample particles based on normalized weights
        resampled_indices = np.random.choice(len(self.particle_cloud), size=len(self.particle_cloud)-15, p=particle_weights) # resamples 285 particles
        resampled_particles = [self.particle_cloud[i] for i in resampled_indices] # extract samples particle objects

        # Define noise ranges
        pos_range = .2 # 20 cm radius around each particle
        head_range = np.pi / 8 # 22.5 degree range in radians

        # Resampled positions and headings for weighted particles with noise
        for i in range(len(resampled_particles)):
            resampled_particles[i].x += rng.uniform(-pos_range, pos_range) # Inject noise to x-coordinate
            resampled_particles[i].y += rng.uniform(-pos_range, pos_range) # Inject noise to y-coordinate
            resampled_particles[i].theta += rng.uniform(-head_range, head_range) # Inject noise to heading
        
        # Make 15 random particles
        n_rand_particles = 15
        rand_location_range = 2 # include random particles within 2 meters of the last known position
        rand_heading_range = np.pi / 4 # 45 degree range in radians

        last_pose = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose) # last know position

        rand_particles = [Particle(x=last_pose[0], y=last_pose[1], theta=last_pose[2], w=1.0/self.n_particles) for _ in range(n_rand_particles)] # create 15 particle object of the last known position

        # Resampled positions and headings for random particles
        rand_x_positions = rng.uniform(low=-rand_location_range, high=rand_location_range, size=n_rand_particles)
        rand_y_positions = rng.uniform(low=-rand_location_range, high=rand_location_range, size=n_rand_particles)
        headings = rng.uniform(low=-rand_heading_range, high=rand_heading_range, size=n_rand_particles)
        
        self.particle_cloud = []  # Clear the current particle cloud

        # Add resamples particles
        for i in range(self.n_particles - n_rand_particles):
            self.particle_cloud.append(Particle(x=resampled_particles[i].x, y=resampled_particles[i].y, theta=resampled_particles[i].theta, w=1.0/self.n_particles))

        for i in range(n_rand_particles):
            self.particle_cloud.append(Particle(x=rand_particles[i].x+rand_x_positions[i], y=rand_particles[i].y+rand_y_positions[i], theta=rand_particles[i].theta+headings[i], w=1.0/self.n_particles))

        return resampled_particles


   def ray_cast(self, ox, oy, angles, max_range):
       """ Sphere-trace rays through the occupancy field.
           ox, oy, angles: 1D arrays, one entry per ray (map frame).
           Returns the expected range for each ray, capped at max_range. """
       res = self.occupancy_field.map.info.resolution
       hit_thresh = 0.5 * res
       min_step = 0.5 * res
       max_steps = 60


       n = len(ox)
       dist = np.zeros(n)
       c, s = np.cos(angles), np.sin(angles)
       active = np.arange(n)                      # indices of rays still marching


       for _ in range(max_steps):
           if active.size == 0:
               break
           px = ox[active] + dist[active] * c[active]
           py = oy[active] + dist[active] * s[active]
           d = self.occupancy_field.get_closest_obstacle_distance(px, py)


           off_map = np.isnan(d)                  # ray left the map: no hit
           hit = ~off_map & (d < hit_thresh)      # ray reached a wall
           dist[active[off_map]] = max_range


           keep = ~(off_map | hit)
           marching = active[keep]


           dist[marching] += d[keep]


           too_far = dist[marching] >= max_range
           dist[marching[too_far]] = max_range
           active = marching[~too_far]


       return np.minimum(dist, max_range)


   def update_particles_with_laser(self, r, theta):
       """ Updates the particle weights in response to the scan data
           r: the distance readings to obstacles
           theta: the angle relative to the robot frame for each corresponding reading
       """
       # TODO: implement this
       #simulated_ranges = self.project scan
       #weights = []
       r = np.asarray(r, dtype=float)
       theta = np.asarray(theta, dtype=float)
       max_range = 5.0          # cap; beyond this the map is less reliable
       beam_step = 10           # use every 10th beam for speed
       noise = 0.15             # expected sensor noise (m); tune this


       r, theta = r[::beam_step], theta[::beam_step]
       valid = np.isfinite(r) & (r > 0.1) & (r < max_range)
       r, theta = r[valid], theta[valid]
       if len(r) == 0 or not self.particle_cloud:
           return


       P, B = len(self.particle_cloud), len(r)
       px = np.array([p.x for p in self.particle_cloud])
       py = np.array([p.y for p in self.particle_cloud])
       pt = np.array([p.theta for p in self.particle_cloud])


       # one ray per (particle, beam), flattened to 1D
       ox = np.repeat(px, B)
       oy = np.repeat(py, B)
       angles = (pt[:, None] + theta[None, :]).ravel()


       expected = self.ray_cast(ox, oy, angles, max_range).reshape(P, B)
       err = expected - r[None, :]


       # Gaussian around the expected range, plus a floor so one bad beam
       # (a person walking by, a glass door) can't zero out a good particle
       p_beam = 0.9 * np.exp(-0.5 * (err / noise) ** 2) + 0.1


       # combine beams in log space so the product doesn't underflow to 0
       log_w = np.sum(np.log(p_beam), axis=1)
       w = np.exp(log_w - log_w.max())


       for p, wi in zip(self.particle_cloud, w):
           p.w = wi
          


   def update_initial_pose(self, msg):
       """ Callback function to handle re-initializing the particle filter based on a pose estimate.
           These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
       xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
       self.initialize_particle_cloud(msg.header.stamp, xy_theta)


   def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                        particle cloud around.  If this input is omitted, the odometry will be used """
        if xy_theta is None:
            xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        self.particle_cloud = []

        # Create number generator
        rng = np.random.default_rng()


        particle_cloud_radius = 1

        x_positions = rng.uniform(low=xy_theta[0]-particle_cloud_radius, high=xy_theta[0]+particle_cloud_radius, size=self.n_particles)
        y_positions = rng.uniform(low=xy_theta[1]-particle_cloud_radius, high=xy_theta[1]+particle_cloud_radius, size=self.n_particles)
        headings = rng.uniform(low=0, high=0, size=self.n_particles)
        for i in range(self.n_particles):
            self.particle_cloud.append(Particle(x=x_positions[i], y=y_positions[i], theta=headings[i], w=1.0/self.n_particles))

        #sample from particles around xy_theta with some noise
        
        self.particle_cloud[0,:] = x_positions
        self.particle_cloud[1,:] = y_positions
        self.particle_cloud[2,:] = headings
        self._particle_cloud[3,:] = 1.0/self.n_particles # initally equal weights for all particles

   def normalize_particles(self):
       """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
       # TODO: implement this
       pass


   def publish_particles(self, timestamp):
       msg = ParticleCloud()
       msg.header.frame_id = self.map_frame
       msg.header.stamp = timestamp
       for p in self.particle_cloud:
           msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
       self.particle_pub.publish(msg)


   def scan_received(self, msg):
       self.last_scan_timestamp = msg.header.stamp
       # we throw away scans until we are done processing the previous scan
       # self.scan_to_process is set to None in the run_loop
       if self.scan_to_process is None:
           self.scan_to_process = msg


def main(args=None):
   rclpy.init()
   n = ParticleFilter()
   rclpy.spin(n)
   rclpy.shutdown()


if __name__ == '__main__':
   main()

