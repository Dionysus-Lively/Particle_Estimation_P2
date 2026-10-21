#!/usr/bin/env python3

""" Particle filter for the robot localization project.

    Estimates the robot's pose (x, y, yaw) in a known map by maintaining a
    cloud of weighted pose hypotheses (particles). Each update cycle:
      1. moves every particle by the odometry change (plus noise),
      2. weights each particle by how well the lidar scan matches the map
         from that particle's pose,
      3. takes the weighted mean as the robot's pose estimate,
      4. resamples so particles concentrate where the weights are high.
"""

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
   """ One hypothesis of the robot's pose in the map frame.

       Attributes:
           x: x-coordinate of the hypothesis in the map frame (m)
           y: y-coordinate of the hypothesis in the map frame (m)
           theta: yaw of the hypothesis in the map frame (rad)
           w: weight of the hypothesis. The class does not keep weights
              normalized; ParticleFilter.normalize_particles does that.
   """

   def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
       """ Construct a new Particle.

           Args:
               x: x-coordinate in the map frame (m)
               y: y-coordinate in the map frame (m)
               theta: yaw in the map frame (rad)
               w: particle weight (not necessarily normalized)
       """
       self.w = w
       self.theta = theta
       self.x = x
       self.y = y

   def as_pose(self):
       """ Convert this particle to a geometry_msgs/Pose.

           Returns:
               Pose with position (x, y, 0) and an orientation quaternion
               representing a pure rotation of theta about the z-axis.
       """
       q = quaternion_from_euler(0, 0, self.theta)
       return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                   orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))


class ParticleFilter(Node):
   """ ROS 2 node that localizes the robot in a known map with a particle filter.

       Attributes:
           base_frame: name of the robot base frame
           map_frame: name of the map frame
           odom_frame: name of the odometry frame
           scan_topic: topic to listen to for lidar scans
           n_particles: number of particles in the filter
           d_thresh: linear movement (m) in x or y before a filter update runs
           a_thresh: angular movement (rad) before a filter update runs
           particle_pub: publisher for the particle cloud (for rviz)
           last_scan_timestamp: stamp of the most recent scan, used to time the
                                map -> odom transform (important when playing bags)
           scan_to_process: the scan run_loop should process next, or None
           occupancy_field: answers "distance to nearest obstacle" queries on the map
           transform_helper: wraps tf2 lookups and conversions
           particle_cloud: list of Particle objects representing the pose belief
           current_odom_xy_theta: odom pose [x, y, theta] at the last filter update
           odom_pose: odom pose (Pose) matching the scan being processed
           robot_pose: current best estimate of the robot pose in the map (Pose)
   """

   def __init__(self):
       """ Set up parameters, publishers, subscribers, helpers, and the
           background thread that runs the filter loop. """
       super().__init__('pf')
       self.base_frame = "base_link"   # the frame of the robot base
       self.map_frame = "map"          # the name of the map coordinate frame
       self.odom_frame = "odom"        # the name of the odometry coordinate frame
       self.scan_topic = "scan"        # the topic where we will get laser scans from

       self.n_particles = 600          # the number of particles to use

       # Only run a (fairly expensive) filter update once the robot has moved
       # this far, so particles aren't re-weighted on nearly identical scans.
       self.d_thresh = 0.2             # meters, in x or y
       self.a_thresh = math.pi/6       # radians (30 degrees)

       # Respond to "2D Pose Estimate" clicks in rviz by re-initializing the cloud
       self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

       # Publish the particle cloud so it can be viewed in rviz
       self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)

       # Listen for lidar scans
       self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

       # Timestamp of the latest scan. When playing a bag, this is the bag's
       # clock, which we use to stamp the map -> odom transform correctly.
       self.last_scan_timestamp = None
       # The scan run_loop should process next (None means nothing waiting)
       self.scan_to_process = None
       # The pose belief: a list of weighted Particle objects
       self.particle_cloud = []

       # Odom pose at the last filter update; empty until the first scan arrives
       self.current_odom_xy_theta = []
       self.occupancy_field = OccupancyField(self)
       self.transform_helper = TFHelper(self)

       # Run the filter loop on its own thread so heavy processing doesn't
       # block ROS callbacks (scans, rviz poses) on the single-threaded executor
       thread = Thread(target=self.loop_wrapper)
       thread.start()
       # Re-broadcast the map -> odom transform at 20 Hz
       self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

   def pub_latest_transform(self):
       """ Broadcast the most recent map -> odom transform.

           The stamp is set slightly in the future (scan time + 0.1 s) so that
           tf lookups by other nodes (e.g. rviz) don't fail for being newer
           than the latest transform.
       """
       if self.last_scan_timestamp is None:
           return
       postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
       self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)

   def loop_wrapper(self):
       """ Call run_loop forever at roughly 10 Hz.

           Runs on a separate thread to work around ROS 2's single-threaded
           executor, which would otherwise block callbacks during processing.
       """
       while True:
           self.run_loop()
           time.sleep(0.1)

   def run_loop(self):
       """ Main filter loop: process the waiting scan, if any.

           Finds the odometry pose that matches the scan's timestamp, then
           either initializes the filter (first scans) or, if the robot has
           moved far enough, runs one full update: odometry motion, laser
           weighting, pose estimate, and resampling. Publishes the particle
           cloud every time a scan is processed.
       """
       if self.scan_to_process is None:
           return
       msg = self.scan_to_process

       # Look up where odometry thought the robot was when the scan was taken
       (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                          self.base_frame,
                                                                          msg.header.stamp)
       if new_pose is None:
           # We couldn't get the robot pose for the scan's timestamp
           if delta_t is not None and delta_t < Duration(seconds=0.0):
               # The scan is older than any transform we have, so we will
               # never get it; drop the scan
               self.scan_to_process = None
           return

       # Convert the scan into (range, angle) pairs in the robot's base frame
       (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
       print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
       # Clear the current scan so scan_received can queue the next one
       self.scan_to_process = None

       self.odom_pose = new_pose
       new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
       print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

       if not self.current_odom_xy_theta:
           # First scan: just remember where odometry says we are
           self.current_odom_xy_theta = new_odom_xy_theta
       elif not self.particle_cloud:
           # Transforms are available now, so we can build the initial cloud
           self.initialize_particle_cloud(msg.header.stamp)
       elif self.moved_far_enough_to_update(new_odom_xy_theta):
           # Robot has moved enough to be worth a full filter update
           self.update_particles_with_odom()            # predict: move particles
           self.update_particles_with_laser(r, theta)   # correct: weight by scan
           self.update_robot_pose()                     # estimate: weighted mean
           self.resample_particles()                    # focus on likely poses
       # Publish particles so rviz can display them
       self.publish_particles(msg.header.stamp)

   def moved_far_enough_to_update(self, new_odom_xy_theta):
       """ Check whether the robot has moved enough since the last update.

           Args:
               new_odom_xy_theta: current odom pose as [x, y, theta]

           Returns:
               True if the change in x or y exceeds d_thresh, or the change
               in heading exceeds a_thresh.
       """
       return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
              math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
              math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh

   def update_robot_pose(self):
        """ Estimate the robot's pose from the particle cloud.

            Uses the weighted mean of the particles' positions and a weighted
            circular mean of their headings. Stores the result in
            self.robot_pose and updates the map -> odom transform so the
            robot appears in the right place in the map.
        """
        self.normalize_particles()

        # Weighted average position
        w = np.array([p.w for p in self.particle_cloud])
        x = np.sum(w * np.array([p.x for p in self.particle_cloud]))
        y = np.sum(w * np.array([p.y for p in self.particle_cloud]))

        # Average angles via sin/cos so that +179° and -179° average to 180°, not 0°
        th = np.array([p.theta for p in self.particle_cloud])
        theta = math.atan2(np.sum(w * np.sin(th)), np.sum(w * np.cos(th)))

        self.robot_pose = Particle(x=x, y=y, theta=theta).as_pose()

        # map -> odom is whatever correction makes odom's pose line up with our estimate
        if hasattr(self, 'odom_pose'):
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose, self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

   def update_particles_with_odom(self):
       """ Move every particle by the robot's motion since the last update.

           The odometry change is converted into the robot's own frame
           (forward, sideways, turn) using the old odom heading. Each particle
           then applies that same motion relative to its own heading, with
           Gaussian noise that grows with the size of the motion, so particles
           spread out to reflect odometry uncertainty.
       """
       new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
       # Compute the change in x, y, theta (odom frame) since the last update
       if self.current_odom_xy_theta:
           old_odom_xy_theta = self.current_odom_xy_theta
           delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                    new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                    new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

           self.current_odom_xy_theta = new_odom_xy_theta
       else:
           # No previous pose to compare against; just record this one
           self.current_odom_xy_theta = new_odom_xy_theta
           return

       # Rotate the odom-frame displacement into the robot's frame at the old
       # pose: fwd is motion along the robot's heading, angle is sideways (left)
       old_th = old_odom_xy_theta[2]
       fwd  =  math.cos(old_th) * delta[0] + math.sin(old_th) * delta[1]
       angle = -math.sin(old_th) * delta[0] + math.cos(old_th) * delta[1]
       dth  = math.atan2(math.sin(delta[2]), math.cos(delta[2]))   # wrap turn to [-pi, pi]

       rng = np.random.default_rng()
       for p in self.particle_cloud:
           # Noise: 2 cm / 0.02 rad baseline plus 10% of the motion
           f = fwd   + rng.normal(0, 0.02 + 0.1 * abs(fwd))
           l = angle + rng.normal(0, 0.02 + 0.1 * abs(angle))
           t = dth   + rng.normal(0, 0.02 + 0.1 * abs(dth))

           # Apply the robot-frame motion in this particle's own frame
           p.x += f * math.cos(p.theta) - l * math.sin(p.theta)
           p.y += f * math.sin(p.theta) + l * math.cos(p.theta)
           p.theta += t

   def resample_particles(self):
        """ Draw a new particle cloud according to the current weights.

            Most of the new cloud (n_particles - 30) is drawn with replacement
            in proportion to weight, then jittered (±0.5 m in x and y,
            ±22.5° in heading) so duplicates spread out. The remaining 30
            particles are placed near the current pose estimate (±0.1 m,
            ±45°) to help the filter recover if the cloud drifts. All new
            particles get equal weight.

            Requires self.robot_pose, so update_robot_pose must run first.
        """
        # Make sure the distribution is normalized
        self.normalize_particles()

        rng = np.random.default_rng()

        # Weights as a probability vector (renormalized to guard against rounding)
        particle_weights = np.array([particle.w for particle in self.particle_cloud])
        particle_weights /= np.sum(particle_weights)

        n_rand_particles = 30   # particles seeded around the pose estimate

        # Pick indices with probability proportional to weight (570 of 600 by default)
        resampled_indices = np.random.choice(len(self.particle_cloud), size=len(self.particle_cloud)-n_rand_particles, p=particle_weights)
        resampled_particles = [self.particle_cloud[i] for i in resampled_indices]

        # Jitter applied to resampled particles
        pos_range = .5           # ±0.5 m in x and in y
        head_range = np.pi / 8   # ±22.5 degrees

        # Copy each chosen particle with uniform noise added, so repeated picks
        # of the same particle become distinct hypotheses
        temp_particles = [
            Particle(x=resampled_particles[i].x + rng.uniform(-pos_range, pos_range),
                     y=resampled_particles[i].y + rng.uniform(-pos_range, pos_range),
                     theta=resampled_particles[i].theta + rng.uniform(-head_range, head_range),
                     w=1.0/self.n_particles)
            for i in range(len(resampled_particles))]

        resampled_particles = temp_particles

        # Spread for the 30 extra particles around the pose estimate
        rand_location_range = 0.1        # ±0.1 m in x and in y
        rand_heading_range = np.pi / 4   # ±45 degrees

        last_pose = self.transform_helper.convert_pose_to_xy_and_theta(self.robot_pose)   # current pose estimate

        # Start each extra particle exactly at the pose estimate...
        rand_particles = [Particle(x=last_pose[0], y=last_pose[1], theta=last_pose[2], w=1.0/self.n_particles) for _ in range(n_rand_particles)]

        # ...then draw the offsets that will be added to them
        rand_x_positions = rng.uniform(low=-rand_location_range, high=rand_location_range, size=n_rand_particles)
        rand_y_positions = rng.uniform(low=-rand_location_range, high=rand_location_range, size=n_rand_particles)
        headings = rng.uniform(low=-rand_heading_range, high=rand_heading_range, size=n_rand_particles)

        # Rebuild the cloud from scratch
        self.particle_cloud = []

        # Add the resampled particles (note: assumes the old cloud had n_particles)
        for i in range(self.n_particles - n_rand_particles):
            self.particle_cloud.append(Particle(x=resampled_particles[i].x, y=resampled_particles[i].y, theta=resampled_particles[i].theta, w=1.0/self.n_particles))

        # Add the extra particles near the pose estimate
        for i in range(n_rand_particles):
            self.particle_cloud.append(Particle(x=rand_particles[i].x+rand_x_positions[i], y=rand_particles[i].y+rand_y_positions[i], theta=rand_particles[i].theta+headings[i], w=1.0/self.n_particles))

   def ray_cast(self, ox, oy, angles, max_range):
       """ Find how far each ray travels through the map before hitting a wall.

           Uses sphere tracing: at each point, the distance to the nearest
           obstacle is a safe step size, since nothing can be closer than that
           in any direction. Rays keep stepping until they get within one map
           cell of an obstacle, leave the map, or pass max_range. All rays are
           processed together with numpy for speed.

           Args:
               ox, oy: 1D arrays of ray start points in the map frame (m)
               angles: 1D array of ray directions in the map frame (rad)
               max_range: maximum distance to report (m)

           Returns:
               1D array of expected ranges, one per ray, capped at max_range.
       """
       res = self.occupancy_field.map.info.resolution
       hit_thresh = 1.0 * res   # within one cell of an obstacle counts as a hit
       min_step = 0.5 * res     # (currently unused)
       max_steps = 150          # safety limit on iterations

       n = len(ox)
       dist = np.zeros(n)                         # distance traveled by each ray
       c, s = np.cos(angles), np.sin(angles)      # ray direction unit vectors
       active = np.arange(n)                      # indices of rays still marching

       for _ in range(max_steps):
           if active.size == 0:
               break
           # Current tip of each active ray, and its clearance to the nearest obstacle
           px = ox[active] + dist[active] * c[active]
           py = oy[active] + dist[active] * s[active]
           d = self.occupancy_field.get_closest_obstacle_distance(px, py)

           off_map = np.isnan(d)                  # ray left the map: no hit
           hit = ~off_map & (d < hit_thresh)      # ray reached a wall
           dist[active[hit]] += d[hit]            # close the small remaining gap
           dist[active[off_map]] = max_range      # treat leaving the map as no return

           # Rays that neither hit nor left the map step forward by their clearance
           keep = ~(off_map | hit)
           marching = active[keep]

           dist[marching] += d[keep]

           # Stop rays that have gone past max_range
           too_far = dist[marching] >= max_range
           dist[marching[too_far]] = max_range
           active = marching[~too_far]

       return np.minimum(dist, max_range)

   def update_particles_with_laser(self, r, theta):
       """ Re-weight particles by how well the scan matches the map from each pose.

           For each particle and each (subsampled, valid) beam, ray-casts the
           range the lidar *should* read if the robot were at that particle's
           pose, and compares it to the actual reading. Weights are scaled so
           the best particle has weight 1.0; normalize_particles makes them
           sum to 1 later.

           Args:
               r: measured ranges (m)
               theta: beam angles in the robot frame (rad), one per range
       """
       r = np.asarray(r, dtype=float)
       theta = np.asarray(theta, dtype=float)
       max_range = 5.0     # ignore readings beyond this; the map is less reliable far away
       beam_step = 10      # use every 10th beam for speed
       noise = 0.3         # expected sensor noise std dev (m); tune this

       # Subsample beams and drop invalid ones (inf/nan, too close, too far)
       r, theta = r[::beam_step], theta[::beam_step]
       valid = np.isfinite(r) & (r > 0.1) & (r < max_range)
       r, theta = r[valid], theta[valid]
       if len(r) == 0 or not self.particle_cloud:
           # Nothing usable to compare against; leave weights unchanged
           return

       # P particles x B beams
       P, B = len(self.particle_cloud), len(r)
       px = np.array([p.x for p in self.particle_cloud])
       py = np.array([p.y for p in self.particle_cloud])
       pt = np.array([p.theta for p in self.particle_cloud])

       # One ray per (particle, beam), flattened to 1D for ray_cast.
       # Beam direction in the map = particle heading + beam angle.
       ox = np.repeat(px, B)
       oy = np.repeat(py, B)
       angles = (pt[:, None] + theta[None, :]).ravel()

       # Expected range for every (particle, beam) and its error vs. the real scan
       expected = self.ray_cast(ox, oy, angles, max_range).reshape(P, B)
       err = expected - r[None, :]

       # Gaussian around the expected range, plus a floor so one bad beam
       # (a person walking by, a glass door) can't zero out a good particle
       p_beam = 0.9 * np.exp(-0.5 * (err / noise) ** 2) + 0.1

       # Combine beams in log space so the product doesn't underflow to 0.
       # Dividing by 3 flattens the distribution so the filter doesn't become
       # overconfident (beams aren't truly independent).
       log_w = np.sum(np.log(p_beam), axis=1)/3.0
       # Subtract the max before exponentiating so the best particle gets 1.0
       w = np.exp(log_w - log_w.max())

       for p, wi in zip(self.particle_cloud, w):
           p.w = wi

   def update_initial_pose(self, msg):
       """ Re-initialize the filter around a pose estimate.

           Called when a pose is published on /initialpose, e.g. from the
           "2D Pose Estimate" tool in rviz.

           Args:
               msg: PoseWithCovarianceStamped with the estimated pose
       """
       xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
       self.initialize_particle_cloud(msg.header.stamp, xy_theta)

   def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Create a fresh particle cloud around a starting pose.

            Particles are spread uniformly within ±1 m in x and y and ±45°
            in heading of the starting pose, all with equal weight.

            Args:
                timestamp: stamp of the triggering message (currently unused)
                xy_theta: [x, y, theta] to center the cloud on. If omitted,
                          the current odometry pose is used.
        """
        if xy_theta is None:
            xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        self.particle_cloud = []

        rng = np.random.default_rng()

        particle_cloud_radius = 1   # meters; half-width of the square the cloud fills

        # Uniform positions and headings around the starting pose
        x_positions = rng.uniform(low=xy_theta[0]-particle_cloud_radius, high=xy_theta[0]+particle_cloud_radius, size=self.n_particles)
        y_positions = rng.uniform(low=xy_theta[1]-particle_cloud_radius, high=xy_theta[1]+particle_cloud_radius, size=self.n_particles)
        headings = rng.uniform(low=xy_theta[2]-np.pi/4, high=xy_theta[2]+np.pi/4, size=self.n_particles)

        for i in range(self.n_particles):
            self.particle_cloud.append(Particle(x=x_positions[i], y=y_positions[i], theta=headings[i], w=1.0/self.n_particles))

   def normalize_particles(self):
       """ Scale particle weights so they sum to 1.

           If every weight is 0, the weights are left unchanged to avoid
           dividing by zero.
       """
       total = sum(p.w for p in self.particle_cloud)
       if total > 0:
           for p in self.particle_cloud:
               p.w /= total

   def publish_particles(self, timestamp):
       """ Publish the particle cloud as a nav2_msgs/ParticleCloud for rviz.

           Args:
               timestamp: stamp to put on the message (the scan's stamp)
       """
       msg = ParticleCloud()
       msg.header.frame_id = self.map_frame
       msg.header.stamp = timestamp
       for p in self.particle_cloud:
           msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
       self.particle_pub.publish(msg)

   def scan_received(self, msg):
       """ Lidar callback: queue the scan for run_loop if it's free.

           Always records the timestamp (used for the map -> odom transform),
           but drops the scan if the previous one is still being processed,
           so the filter always works on recent data instead of a backlog.

           Args:
               msg: the incoming LaserScan
       """
       self.last_scan_timestamp = msg.header.stamp
       # run_loop sets scan_to_process back to None once it has taken the scan
       if self.scan_to_process is None:
           self.scan_to_process = msg


# ---------------------------------------------------------------------------
# Unit tests
#
# These build a ParticleFilter without starting ROS (no node, thread, or
# timers) and swap in small fakes for the map and transform helper, so each
# test has a known correct answer. Run with:
#     python3 -m pytest pf.py -v      or      python3 -m unittest pf -v
# ---------------------------------------------------------------------------

import unittest
from unittest import mock


class _FakeMapInfo:
    """ Stand-in for the map metadata; only resolution is used. """
    resolution = 0.05

class _FakeMap:
    """ Stand-in for the map message, exposing .info.resolution. """
    info = _FakeMapInfo()

class _FakeField:
    """ Fake occupancy field: a single wall along x = 2.
        The map spans [-10, 10] in x and y; points outside return nan. """
    map = _FakeMap()
    def get_closest_obstacle_distance(self, x, y):
        """ Distance from each (x, y) to the wall, or nan if off the map. """
        x, y = np.asarray(x, float), np.asarray(y, float)
        d = np.abs(2.0 - x)
        d[(np.abs(x) > 10) | (np.abs(y) > 10)] = np.nan
        return d

class _FakeTF:
    """ Fake TFHelper with only the two methods the tested code calls. """
    def convert_pose_to_xy_and_theta(self, pose):
        """ Pose -> [x, y, yaw], extracting yaw from the quaternion. """
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        return [pose.position.x, pose.position.y, yaw]
    def fix_map_to_odom_transform(self, robot_pose, odom_pose):
        """ No-op: tests don't need a real transform. """
        pass

class _ZeroNoiseRng:
    """ Replaces numpy's RNG so motion-model tests are deterministic. """
    def normal(self, loc, scale):
        return 0.0


def _make_pf(n=600):
    """ A ParticleFilter with no ROS node, thread, or timers behind it. """
    pf = ParticleFilter.__new__(ParticleFilter)   # skip __init__ (no ROS)
    pf.n_particles = n
    pf.particle_cloud = []
    pf.d_thresh = 0.2
    pf.a_thresh = math.pi / 6
    pf.occupancy_field = _FakeField()
    pf.transform_helper = _FakeTF()
    pf.odom_pose = Particle().as_pose()
    pf.current_odom_xy_theta = []
    return pf


class TestParticleFilter(unittest.TestCase):
    """ Unit tests for the ParticleFilter's math, independent of ROS. """

    # ---------- normalize_particles ----------
    def test_normalize_sums_to_one(self):
        """ Weights 1, 2, 7 become 0.1, 0.2, 0.7. """
        pf = _make_pf()
        pf.particle_cloud = [Particle(w=w) for w in (1.0, 2.0, 7.0)]
        pf.normalize_particles()
        self.assertAlmostEqual(sum(p.w for p in pf.particle_cloud), 1.0)
        self.assertAlmostEqual(pf.particle_cloud[2].w, 0.7)

    def test_normalize_all_zero_does_not_crash(self):
        """ All-zero weights are left alone instead of dividing by zero. """
        pf = _make_pf()
        pf.particle_cloud = [Particle(w=0.0) for _ in range(5)]
        pf.normalize_particles()
        self.assertTrue(all(p.w == 0.0 for p in pf.particle_cloud))

    # ---------- update_robot_pose ----------
    def test_robot_pose_is_weighted_mean(self):
        """ Pose estimate is the weight-averaged position. """
        pf = _make_pf()
        pf.particle_cloud = [Particle(0, 0, 0, w=1), Particle(4, 2, 0, w=3)]
        pf.update_robot_pose()
        x, y, th = pf.transform_helper.convert_pose_to_xy_and_theta(pf.robot_pose)
        self.assertAlmostEqual(x, 3.0)
        self.assertAlmostEqual(y, 1.5)

    def test_robot_pose_heading_wraps_around_pi(self):
        """ +179° and -179° average to 180°, not 0°. """
        pf = _make_pf()
        a = math.radians(179)
        pf.particle_cloud = [Particle(theta=a), Particle(theta=-a)]
        pf.update_robot_pose()
        th = pf.transform_helper.convert_pose_to_xy_and_theta(pf.robot_pose)[2]
        self.assertAlmostEqual(abs(th), math.pi, places=5)   # not 0

    # ---------- moved_far_enough_to_update ----------
    def test_moved_far_enough(self):
        """ Small moves don't trigger an update; crossing any threshold does. """
        pf = _make_pf()
        pf.current_odom_xy_theta = [0.0, 0.0, 0.0]
        self.assertFalse(pf.moved_far_enough_to_update([0.1, 0.1, 0.1]))
        self.assertTrue(pf.moved_far_enough_to_update([0.3, 0.0, 0.0]))
        self.assertTrue(pf.moved_far_enough_to_update([0.0, -0.3, 0.0]))
        self.assertTrue(pf.moved_far_enough_to_update([0.0, 0.0, math.pi / 4]))

    # ---------- update_particles_with_odom ----------
    def test_odom_forward_motion_follows_each_particles_heading(self):
        """ Driving forward moves each particle along its own heading. """
        pf = _make_pf()
        # robot faces +y in odom and drives 1 m forward
        pf.current_odom_xy_theta = [0.0, 0.0, math.pi / 2]
        pf.odom_pose = Particle(0, 1, math.pi / 2).as_pose()
        pf.particle_cloud = [Particle(0, 0, 0), Particle(5, 5, math.pi)]
        with mock.patch.object(np.random, "default_rng", return_value=_ZeroNoiseRng()):
            pf.update_particles_with_odom()
        a, b = pf.particle_cloud
        self.assertAlmostEqual(a.x, 1.0); self.assertAlmostEqual(a.y, 0.0)
        self.assertAlmostEqual(b.x, 4.0); self.assertAlmostEqual(b.y, 5.0)

    def test_odom_rotation_is_wrapped(self):
        """ A turn across ±pi is applied as the short way around. """
        pf = _make_pf()
        # 3.0 -> -3.0 rad is a small +0.28 rad turn, not a -6 rad spin
        pf.current_odom_xy_theta = [0.0, 0.0, 3.0]
        pf.odom_pose = Particle(0, 0, -3.0).as_pose()
        pf.particle_cloud = [Particle(0, 0, 0)]
        with mock.patch.object(np.random, "default_rng", return_value=_ZeroNoiseRng()):
            pf.update_particles_with_odom()
        self.assertAlmostEqual(pf.particle_cloud[0].theta, 2 * math.pi - 6.0, places=5)

    # ---------- ray_cast ----------
    def test_ray_cast_known_distances(self):
        """ Rays toward, angled at, away from, and off the map return expected ranges. """
        pf = _make_pf()
        ox = np.array([0.0, 0.0, 0.0, 0.0])
        oy = np.array([0.0, 0.0, 0.0, 9.9])
        angles = np.array([0.0, math.pi / 3, math.pi, math.pi / 2])
        d = pf.ray_cast(ox, oy, angles, max_range=5.0)
        self.assertAlmostEqual(d[0], 2.0, delta=0.1)   # straight at the wall
        self.assertAlmostEqual(d[1], 4.0, delta=0.15)  # 60 deg: 2 / cos(60)
        self.assertEqual(d[2], 5.0)                     # facing away: capped
        self.assertEqual(d[3], 5.0)                     # leaves the map: capped

    # ---------- update_particles_with_laser ----------
    def test_laser_prefers_true_pose(self):
        """ The particle at the true pose gets the highest weight. """
        pf = _make_pf()
        theta = np.linspace(-0.6, 0.6, 100)
        r = 2.0 / np.cos(theta)                         # scan from (0, 0, 0)
        pf.particle_cloud = [Particle(0, 0, 0), Particle(1, 0, 0), Particle(0, 0, 0.5)]
        pf.update_particles_with_laser(r, theta)
        w = [p.w for p in pf.particle_cloud]
        self.assertAlmostEqual(w[0], 1.0)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[0], w[2])

    def test_laser_ignores_scan_with_no_valid_beams(self):
        """ A scan with no usable beams leaves the weights unchanged. """
        pf = _make_pf()
        pf.particle_cloud = [Particle(w=0.25) for _ in range(4)]
        theta = np.linspace(-1, 1, 50)
        pf.update_particles_with_laser(np.full(50, np.inf), theta)
        self.assertTrue(all(p.w == 0.25 for p in pf.particle_cloud))

    # ---------- initialize_particle_cloud ----------
    def test_initialize_cloud(self):
        """ Cloud has n particles within ±1 m and ±45° of the start, equal weights. """
        pf = _make_pf(n=600)
        pf.initialize_particle_cloud(None, [3.0, -2.0, 0.5])
        self.assertEqual(len(pf.particle_cloud), 600)
        for p in pf.particle_cloud:
            self.assertTrue(2.0 <= p.x <= 4.0)
            self.assertTrue(-3.0 <= p.y <= -1.0)
            self.assertTrue(0.5 - math.pi / 4 <= p.theta <= 0.5 + math.pi / 4)
            self.assertAlmostEqual(p.w, 1 / 600)

    # ---------- resample_particles ----------
    def test_resample_collapses_onto_heavy_particle(self):
        """ If one particle holds all the weight, the new cloud clusters around it. """
        np.random.seed(0)
        pf = _make_pf(n=100)
        pf.particle_cloud = [Particle(-5, -5, 0, w=0.0) for _ in range(99)]
        pf.particle_cloud.append(Particle(5, 5, 0, w=1.0))
        pf.robot_pose = Particle(5, 5, 0).as_pose()
        pf.resample_particles()
        self.assertEqual(len(pf.particle_cloud), 100)
        for p in pf.particle_cloud:
            self.assertAlmostEqual(p.w, 1 / 100)
            self.assertLessEqual(abs(p.x - 5), 0.5 + 1e-9)
            self.assertLessEqual(abs(p.y - 5), 0.5 + 1e-9)


def main(args=None):
   """ Start the particle filter node and spin until shutdown. """
   rclpy.init()
   n = ParticleFilter()
   rclpy.spin(n)
   rclpy.shutdown()


if __name__ == '__main__':
   main()