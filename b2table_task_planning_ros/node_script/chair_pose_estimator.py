from collections import deque

import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from detic_ros.msg import SegmentationInfo
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
from sensor_msgs.msg import PointCloud2
from sklearn.decomposition import PCA
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy2quaternion


class AverageQueue:
    def __init__(self, n_elem: int):
        self.n_elem = n_elem
        self.queue = deque(maxlen=n_elem)

    def put(self, item: np.ndarray):
        self.queue.append(item)

    def get(self):
        if len(self.queue) < self.n_elem:
            return None
        data = np.array(self.queue)  # shape: (n_elem, array_dim)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        within_2sigma = np.all(np.abs(data - mean) <= 2 * std, axis=1)
        filtered_data = data[within_2sigma]

        if len(filtered_data) == 0:
            return None

        return np.mean(filtered_data, axis=0)


class RedChairPoseEstimator:
    def __init__(
        self,
        segm_topic: str = "~segmentation_info",
        pc_topic: str = "~input_pc",
        out_topic: str = "~chair_boxes",
        n_average: int = 3,
    ):
        segm_sub = message_filters.Subscriber(segm_topic, SegmentationInfo)
        pc_sub = message_filters.Subscriber(pc_topic, PointCloud2)

        sync = message_filters.ApproximateTimeSynchronizer(
            [segm_sub, pc_sub], queue_size=10, slop=1.0, allow_headerless=True
        )
        sync.registerCallback(self.callback)
        self.est_fridge_pub = rospy.Publisher(out_topic, BoundingBoxArray, latch=True, queue_size=1)
        if n_average > 1:
            self.queue = AverageQueue(n_average)
        else:
            self.queue = None

    def callback(self, segm: SegmentationInfo, pc: PointCloud2):
        bridge = CvBridge()
        seg_image = bridge.imgmsg_to_cv2(segm.segmentation, desired_encoding="passthrough")
        n_label = np.max(seg_image)
        print(f"{n_label} of chairs are detected")
        points = pointcloud2_to_xyz_array(pc, remove_nans=False).reshape(-1, 3)

        bbox_array = BoundingBoxArray()
        bbox_array.header = pc.header
        for i_label in range(n_label):
            mask = seg_image == i_label + 1
            # TODO: dilate??
            points_filtered = points[mask.reshape(-1)]
            print(f"len of points: {len(points_filtered)} out of {len(points)}")

            # remove points that has nan element
            points_filtered = points_filtered[~np.isnan(points_filtered).any(axis=1)]

            z_max = 0.9
            point_higher = points_filtered[points_filtered[:, 2] > z_max - 0.1]
            points2d_higher = point_higher[:, :2]  # project to 2D
            if len(points2d_higher) < 100:
                rospy.logwarn(f"Too few points for label {i_label}")
                continue
            points2d_higher_mean = np.mean(points2d_higher, axis=0)

            points_lower = points_filtered[points_filtered[:, 2] < 0.6]
            points_lower = points_lower[points_lower[:, 2] > 0.3]
            points2d_lower = points_lower[:, :2]
            points2d_lower_mean = np.mean(points2d_lower, axis=0)
            x_direction_rough = np.hstack(
                [points2d_lower_mean - points2d_higher_mean, 0]
            )  # from the chair's backrest to seat

            # use pca to determine the direction of the chair
            pca = PCA(n_components=2)
            pca.fit(points2d_higher)
            y_direction = np.hstack([pca.components_[0], 0.0])

            # the cross product of x and y must be pointing up, otherwise, we need to flip the y_direction
            direction = np.cross(x_direction_rough, y_direction)
            if direction[2] < 0:
                y_direction = -y_direction

            yaw = np.arctan2(y_direction[1], y_direction[0]) - 0.5 * np.pi
            quat_wxyz = rpy2quaternion([yaw, 0, 0])

            # calculate the center point
            box_extent = [0.5, 0.5, z_max]
            co = Coordinates(np.hstack([points2d_higher_mean, z_max * 0.5]), quat_wxyz)
            co.translate([0.5 * box_extent[0], 0, 0])

            # pub the result
            bbox = BoundingBox()
            bbox.header = pc.header
            pos_center = co.worldpos()
            (bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z) = pos_center
            (
                bbox.pose.orientation.w,
                bbox.pose.orientation.x,
                bbox.pose.orientation.y,
                bbox.pose.orientation.z,
            ) = quat_wxyz
            (bbox.dimensions.x, bbox.dimensions.y, bbox.dimensions.z) = box_extent
            bbox_array.boxes.append(bbox)
        self.est_fridge_pub.publish(bbox_array)
        print("published")


if __name__ == "__main__":
    rospy.init_node("chair_pose_estimator")
    # activate_detic()
    segm_topic = "/local/detic_segmentor/segmentation_info"
    pc_topic = "/local/tf_transform/output"
    out_topic = "/local/chair_pose_estimator/chair_boxes"
    estimator = RedChairPoseEstimator(segm_topic=segm_topic, pc_topic=pc_topic, out_topic=out_topic)
    rospy.spin()

