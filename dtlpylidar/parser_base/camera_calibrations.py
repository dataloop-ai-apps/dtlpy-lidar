from . import extrinsic_calibrations


class Intrinsic:
    def __init__(self, fx=0.0, fy=0.0, cx=0.0, cy=0.0, skew=0.0):
        """
        Intrinsic Matrix parameters
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :param skew:
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.skew = skew
        self.intrinsicMatrix = [fx, skew, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    def to_json(self, distortion=None):
        """
        Intrinsic matrix and distortion to dict.
        :param distortion:
        :return:
        """
        if distortion is None:
            distortion = Distortion()
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'skew': self.skew,
            'distortion': distortion.to_json()
        }


class Distortion:
    def __init__(self, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
        """
        Distortion matrix parameters
        :param k1:
        :param k2:
        :param k3:
        :param p1:
        :param p2:
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

    def to_json(self):
        """
        Distortion object to dict
        :return:
        """
        return {
            'k1': self.k1,
            'k2': self.k2,
            'k3': self.k3,
            'p1': self.p1,
            'p2': self.p2
        }


class LidarCameraData:
    def __init__(self, intrinsic: Intrinsic, extrinsic: extrinsic_calibrations.Extrinsic, distortion: Distortion,
                 channel=None, cam_id=0, cam_name=None):
        """
        Lidar camera Object.
        :param intrinsic: Camera intrinsic values
        :param extrinsic: Camera extrinsic values
        :param distortion: Camera Distortion
        :param channel: Camera name
        :param cam_id: Camera Id
        """
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.distortion = distortion
        self.channel = channel
        self.cam_id = cam_id
        self.cam_name = cam_name if cam_name is not None else "cam_{}".format(self.cam_id)

    def to_json(self):
        """
        Camera object to json.
        :return:
        """
        return {
            'name': self.cam_name,
            'id': str(self.cam_id),
            'sensorsData': {
                'extrinsic': self.extrinsic.to_json(translation_key='position'),
                'intrinsicMatrix': self.intrinsic.intrinsicMatrix,
                'intrinsicData': self.intrinsic.to_json(distortion=self.distortion)
            },
            'cameraNumber': self.cam_id,
            'channel': self.channel
        }
