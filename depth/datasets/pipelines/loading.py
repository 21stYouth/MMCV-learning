import json
import mmcv
import numpy as np
import os.path as osp
from PIL import Image
from ..builder import PIPELINES
import matplotlib.pyplot as plt
import random


@PIPELINES.register_module()
class LoadKITTICamIntrinsic(object):
    """Load KITTI intrinsic
    """
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        # raw input
        if 'input' in  results['img_prefix']:
            date = results['filename'].split('/')[-5]
            results['cam_intrinsic'] = results['cam_intrinsic_dict'][date]
        # benchmark test
        else:
            temp = results['filename'].replace('benchmark_test', 'benchmark_test_cam')
            cam_file = temp.replace('png', 'txt')
            results['cam_intrinsic'] = np.loadtxt(cam_file).reshape(3, 3).tolist()
        
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class DepthLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        depth_gt = np.asarray(Image.open(filename),
                              dtype=np.float32) / results['depth_scale']

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape

        results['depth_fields'].append('depth_gt')
        # print()
        # exit()
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class DisparityLoadAnnotations(object):
    """Load annotations for depth estimation.
    It's only for the cityscape dataset. TODO: more general.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        if results.get('camera_prefix', None) is not None:
            camera_filename = osp.join(results['camera_prefix'],
                                       results['cam_info']['cam_info'])
        else:
            camera_filename = results['cam_info']['cam_info']

        with open(camera_filename) as f:
            camera = json.load(f)
        baseline = camera['extrinsic']['baseline']
        focal_length = camera['intrinsic']['fx']

        disparity = (np.asarray(Image.open(filename), dtype=np.float32) -
                     1.) / results['depth_scale']
        NaN = disparity <= 0

        disparity[NaN] = 1
        depth_map = baseline * focal_length / disparity
        depth_map[NaN] = 0

        results['depth_gt'] = depth_map
        results['depth_ori_shape'] = depth_map.shape

        results['depth_fields'].append('depth_gt')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFile_v2(object):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        # print("%%%%%%%%%%%%%%%")
        # print(results)
        # print(type(results))

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)



        rint = random.randint(0, 24230)
        f = open("data/nyu/nyu_train.txt")
        lines = f.readlines()
        filename2 = "data/nyu" + lines[rint].split()[0]
        img_bytes2 = self.file_client.get(filename2)
        img2 = mmcv.imfrombytes(img_bytes2,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img2 = img2.astype(np.float32)
        results['img2'] = img2
        results['img_shape2'] = img2.shape
        results['rint'] = rint
        results['filename2'] = filename2


        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)

        # print("##############################################################################")
        # print("filename =", results['filename'])
        # print("ori_filename =", results['ori_filename'])
        # print(results['img_norm_cfg'])
        # print(type(results['img']))
        # print(results)
        # print(type(results))
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img2)
        # plt.show()
        # exit()


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class DepthLoadAnnotations_v2(object):
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        depth_gt = np.asarray(Image.open(filename),
                              dtype=np.float32) / results['depth_scale']

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape

        results['depth_fields'].append('depth_gt')
        # print()
        # exit()

        f = open("data/nyu/nyu_train.txt")
        lines = f.readlines()
        filename2 = "data/nyu" + lines[results["rint"]].split()[1]
        depth_gt2 = np.asarray(Image.open(filename2),
                              dtype=np.float32) / results['depth_scale']

        results['depth_gt2'] = depth_gt2
        results['depth_ori_shape2'] = depth_gt2.shape



        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # plt.imshow(results['depth_gt'])
        # plt.show()
        # plt.imshow(results['depth_gt2'])
        # plt.show()
        # print(results)


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str