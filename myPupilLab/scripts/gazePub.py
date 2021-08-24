#!/usr/bin/env python
import zmq
import msgpack
import rospy
import std_msgs.msg
from myPupilLab.msg import GazeInfoMono
from myPupilLab.msg import GazeInfoBino
from myPupilLab.msg import GazeInfoBino_Array
# from std_msgs.msg import String

ctx = zmq.Context()
# The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
pupil_remote = ctx.socket(zmq.REQ)

ip = 'localhost'  # If you talk to a different machine use its IP.
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

pupil_remote.connect(f'tcp://{ip}:{port}')

# Request 'SUB_PORT' for reading data
pupil_remote.send_string('SUB_PORT')
sub_port = pupil_remote.recv_string()

# Request 'PUB_PORT' for writing data
pupil_remote.send_string('PUB_PORT')
pub_port = pupil_remote.recv_string()

# Assumes `sub_port` to be set to the current subscription port
subscriber = ctx.socket(zmq.SUB)
subscriber.connect(f'tcp://{ip}:{sub_port}') #sub_port not port
subscriber.subscribe('gaze.')  # receive pupil info, 'gaze.' is also valid but need world camera

pub_gaze_bino = rospy.Publisher('gaze_info_bino',GazeInfoBino,queue_size=10) #Node name, message type
pub_gaze_mono = rospy.Publisher('gaze_info_mono',GazeInfoMono,queue_size=10)
pub_gaze_bino_60 = rospy.Publisher('gaze_array',GazeInfoBino_Array,queue_size=1)

def gaze_parser_bino(message):
	#Parse gazeInfo into a message.
	if len(message) < 1: return -1 
	
	outmsg = GazeInfoBino()

	outmsg.eye_centers_3d_0 = message[b'eye_centers_3d'][0]
	outmsg.eye_centers_3d_1 = message[b'eye_centers_3d'][1]
	outmsg.gaze_normals_3d_0 = message[b'gaze_normals_3d'][0]
	outmsg.gaze_normals_3d_1 = message[b'gaze_normals_3d'][1]
	outmsg.gaze_point_3d = message[b'gaze_point_3d']
	outmsg.norm_pos = message[b'norm_pos']
	outmsg.confidence = message[b'confidence']
	outmsg.timestamp = message[b'timestamp']
	x_pos.append(outmsg.norm_pos[0])
	y_pos.append(outmsg.norm_pos[1])

	return outmsg

def gaze_parser_mono(message):
	#Parse gazeInfo into a message.
	if len(message) < 1: return -1 
	
	outmsg = GazeInfoMono()

	outmsg.eye_center_3d = message[b'eye_center_3d']
	outmsg.gaze_normal_3d = message[b'gaze_normal_3d']
	outmsg.gaze_point_3d = message[b'gaze_point_3d']
	outmsg.norm_pos = message[b'norm_pos']
	outmsg.confidence = message[b'confidence']
	outmsg.timestamp = message[b'timestamp']
	x_pos.append(outmsg.norm_pos[0])
	y_pos.append(outmsg.norm_pos[1])
	return outmsg

def pub_array():
	outmsg = GazeInfoBino_Array()
	outmsg.x = x_pos
	outmsg.y = y_pos
	outmsg.header.stamp = rospy.Time.now()
	print("gazeArrayPub Status: Working!")
	pub_gaze_bino_60.publish(outmsg)


if __name__ == "__main__":
	rospy.loginfo("Starting gaze publisher.")
	#print("Starting gaze publisher")

	rospy.init_node('gazePublisher')

	#print("listening for socket message....")
	x_pos = []
	y_pos = []
	count = 0
	while not rospy.is_shutdown():
		topic, payload = subscriber.recv_multipart()
		message = msgpack.loads(payload)
		if topic == b'gaze.3d.01.': #Binocular gaze datum, only occurs when both eyes confidence is higher than a threshold
			#print(message[b'eye_centers_3d'][b'0'])
			outmsg = gaze_parser_bino(message)
			if not outmsg == -1:
				#rospy.loginfo("Bino")
				#rospy.loginfo(outmsg.confidence)
				#rospy.loginfo(outmsg.norm_pos) 
				pub_gaze_bino.publish(outmsg)
				count +=1
				#print(count)
				if count == 30:	
					pub_array()
					count = 0
					x_pos.clear()
					y_pos.clear()
		elif topic == b'gaze.3d.0.' or topic == b'gaze.3d.1.': #Monocular gaze datum, no prerequisite needed
			outmsg = gaze_parser_mono(message)
			if not outmsg == -1:
				if outmsg.confidence > 0.75:
					#rospy.loginfo("Mono")
					#rospy.loginfo(outmsg.confidence)
					#rospy.loginfo(outmsg.norm_pos) 
					pub_gaze_mono.publish(outmsg)
					count +=1
					#print(count)
					if count == 30:	
						pub_array()
						count = 0
						x_pos.clear()
						y_pos.clear()

