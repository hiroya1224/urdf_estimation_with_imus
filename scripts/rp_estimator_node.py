#!/usr/bin/env python3
# ref https://qiita.com/t_katsumura/items/a83431671a41d9b6358f

import socket
import time
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import base64

import ssl
import asyncio
import websockets
import numpy as np
import time
import json

import rospy
from std_msgs.msg import Empty
from urdf_estimation_with_imus.msg import PoseWithCovAndBingham

# HOME = "/home/leus/sensor-https-server"
HOME = "/home/docker/app/"

class RelativePoseWebsocketNode:
    def __init__(self, imu_info_str):
        # イベントループをメインスレッドで取得
        # self._event_loop = asyncio.get_running_loop()
        imu_info = json.loads(imu_info_str)
        target_imus = imu_info["selected_imus"]

        # ROSノードの初期化
        rospy.init_node('async_subscriber_node', anonymous=False, disable_signals=True)
        rospy.loginfo("Initialize node.")

        # Subscriberの作成
        # rospy.Subscriber("/estimated_relative_pose/imucdfe20__to__imua01730", PoseWithCovAndBingham, self.callback)
        subs_topic = "/estimated_relative_pose/{}__to__{}".format(*target_imus)
        rospy.Subscriber(subs_topic, PoseWithCovAndBingham, self.callback)

        rospy.logwarn("Subscribing {}".format(subs_topic))

        self.reset_estimator = rospy.Publisher("/reset_estimation", Empty, queue_size=1)
        self.num_bins = 100

        self.Avec = self.Amat_to_AvecList(np.eye(4) * 1e-10)
        self.position_density = [
            {"mu": 0, "var": 1e+12}, 
            {"mu": 0, "var": 1e+12}, 
            {"mu": 0, "var": 1e+12}, 
        ]
        self.mode_quat_wxyz_list = [1,0,0,0]

    @staticmethod
    def Amat_to_AvecList(Amat):
        return Amat[np.triu_indices(4)].tolist()

    def callback(self, msg):
        self.Avec = self.Amat_to_AvecList(np.array(msg.rotation_bingham_parameter).reshape(4,4))
        mu = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        Sigma = np.array(msg.position_covariance).reshape(3,3)

        self.mode_quat_wxyz_list = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]

        for i in range(3):
            self.position_density[i]["mu"] = mu[i]
            self.position_density[i]["var"] = Sigma[i, i]

            
    async def handler(self, websocket):
        async for msg in websocket:
            if msg == "reset_estimator":
                self.reset_estimator.publish()
                await asyncio.sleep(1)
            else:
                dataset_dict = {"Avec": self.Avec, "gaussianParams": self.position_density, "modeQuaternion": self.mode_quat_wxyz_list}

                # print(json.dumps(dataset_dict))
                await websocket.send(json.dumps(dataset_dict))
                # await rospy_loop("wss://localhost:8002")


context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile=HOME + 'server.crt', keyfile=HOME + 'server.key')

async def server():
    ## load IMU data
    imu_info = None
    while imu_info is None:
        try:
            with open("imu_info.log", "r") as f:
                imu_info = f.read()
        except:
            # imu_info = json.dumps({"message_type": "imu_info", "selected_imus": ["", ""]})
            imu_info = None
            print("loading")
            time.sleep(1)
    
    node = RelativePoseWebsocketNode(imu_info)
    async with websockets.serve(node.handler, "0.0.0.0", 8002, ssl=context):
        await asyncio.Future()  # run forever

async def main():
    await server()
    # await rospy_loop("wss://localhost:8002")

if __name__=="__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--host', help="IP address of the sensor module")
    # parser.add_argument('--noplot', help="disable plot", action="store_true")
    # args = parser.parse_args()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rospy.loginfo("Finished node.")
        rospy.signal_shutdown('finish')