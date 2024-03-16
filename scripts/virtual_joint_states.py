#!/usr/bin/env python3
import rospy
import numpy as np
import xml.etree.ElementTree as ET
from sensor_msgs.msg import JointState


class SinusoidalJointStatePublisher:
    def __init__(self, robot_description, publish_hz,
                 amp_limit=np.pi/2,
                 freq_minlim=1.0, freq_maxlim=3.0):
        self.joints = self.parse_joint_names_from_robot_description(robot_description)
        self.joint_names = [j.attrib["name"] for j in self.joints]
        self.number_of_joints = len(self.joints)
        self.sinusoidal_params = self.get_random_sinusoidal_params(self.number_of_joints,
                                                                   amp_limit=amp_limit,
                                                                   freq_minlim=freq_minlim,
                                                                   freq_maxlim=freq_maxlim)

        self.publisher = rospy.Publisher("/joint_states", JointState, queue_size=1)
        self.publish_hz = publish_hz
        self.publish_dt = 1. / publish_hz
        self.rate = rospy.Rate(publish_hz)
    

    @staticmethod
    def get_random_sinusoidal_params(length, amp_limit=np.pi/2,
                                     freq_minlim=1.0,
                                     freq_maxlim=3.0):
        assert not freq_minlim > freq_maxlim

        amplitudes = np.random.rand(length) * amp_limit
        phases = np.random.rand(length) * 2*np.pi
        freqs = freq_minlim + np.random.rand(length) * (freq_maxlim - freq_minlim)

        return amplitudes, phases, freqs


    @staticmethod
    def parse_joint_names_from_robot_description(robot_description):
        root = ET.fromstring(robot_description)
        alljoints = root.findall("joint")
        joints = [j for j in alljoints if j.attrib["type"] == "revolute"]
        return joints
    

    def publish_jointstates(self):
        # dt = self.publish_dt

        msg = JointState()
        now = rospy.Time.now()
        T = now.to_sec()
        msg.header.stamp = now
        
        ## set random sinusoidal joint states
        msg.name = self.joint_names
        omega = lambda f: 2*np.pi*f
        msg.position = [amp * np.sin(omega(f)*T + phs) for amp, phs, f in zip(*self.sinusoidal_params)]
        msg.velocity = [omega(f) * amp * np.cos(omega(f)*T + phs) for amp, phs, f in zip(*self.sinusoidal_params)]
        msg.effort = [-omega(f)**2 * amp * np.sin(omega(f)*T + phs) for amp, phs, f in zip(*self.sinusoidal_params)]

        ## publish and sleep
        self.publisher.publish(msg)
        self.rate.sleep()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxf", default=3.0, type=float, help="swing frequency [Hz]")
    parser.add_argument("--minf", default=0.5, type=float, help="swing frequency [Hz]")
    parser.add_argument("--amplim", default=1.57, type=float, help="swing frequency [Hz]")
    parser.add_argument("--pubfreq", default=200.0, type=float, help="publish frequency [Hz]")
    parser.add_argument("--seed", default=0, type=int, help="publish frequency [Hz]")

    parser.add_argument("__name", default="", nargs="?", help="for roslaunch")
    parser.add_argument("__log", default="", nargs="?", help="for roslaunch")
    args = parser.parse_args()

    max_swing_freq = args.maxf
    min_swing_freq = args.minf
    amplim = args.amplim
    pubfreq = args.pubfreq
    random_seed = args.seed

    np.random.seed(random_seed)


    rospy.init_node('sinusoidal_jointstates_publisher', anonymous=False)

    rospy.logwarn("swing_freq = {} <= f <= {} [Hz]".format(min_swing_freq, max_swing_freq))
    rospy.logwarn("amplitude limit = {} [Hz]".format(amplim))
    rospy.logwarn("publish freq = {} [Hz]".format(pubfreq))
    rospy.logwarn("random seed = {}".format(random_seed))

    robot_description = rospy.get_param("/robot_description")

    jspub = SinusoidalJointStatePublisher(robot_description, pubfreq,
                                          amp_limit=amplim,
                                          freq_maxlim=max_swing_freq,
                                          freq_minlim=min_swing_freq)
    
    ## publish
    rospy.logwarn("publishing...")
    while not rospy.is_shutdown():
        jspub.publish_jointstates()
    rospy.spin()