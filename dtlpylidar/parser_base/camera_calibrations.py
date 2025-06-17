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
    def __init__(self, **kwargs):
        """
        Distortion parameters. For example:\n
        Radial distortion coefficients: k1, k2, k3, k4, k5, k6, k7, k8.\n
        Tangential distortion coefficients: p1, p2.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        """
        Distortion object to dict
        :return:
        """
        return self.__dict__


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
