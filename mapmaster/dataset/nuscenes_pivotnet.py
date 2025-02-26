# 导入所需库和包
# Import necessary libraries and packages
import os
import numpy as np
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset

class NuScenesMapDataset(Dataset):
    """
    NuScenesMapDataset类继承自torch.utils.data.Dataset，用于加载和预处理NuScenes地图数据集

    The NuScenesMapDataset class inherits from torch.utils.data.Dataset, used for loading and preprocessing the NuScenes map dataset.

    属性/Attributes
    ----------
    img_key_list : list
        包含图像键名的列表，用于标识和提取所需的视图图像
        List of image key names for identifying and extracting required view images
    map_conf : dict
        地图相关配置信息
        Configuration information related to maps
    transforms : function
        数据预处理变换函数
        Data preprocessing transformation function
    tokens : list
        数据切片标记的列表，用于索引各个数据样本
        List of data slice tokens for indexing different data samples
    """

    def __init__(self, img_key_list, map_conf, transforms, data_split="training"):
        super().__init__()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        
        self.ego_size = map_conf["ego_size"]
        self.mask_key = map_conf["mask_key"]
        self.nusc_root = map_conf["nusc_root"]
        self.anno_root = map_conf["anno_root"]
        self.split_dir = map_conf["split_dir"]        # instance_mask/instance_mask8
        
        self.split_mode = 'train' if data_split == "training" else 'val'
        split_path = os.path.join(self.split_dir, f'{self.split_mode}.txt')
        self.tokens = [token.strip() for token in open(split_path).readlines()]
        self.transforms = transforms

    def __getitem__(self, idx: int):
        """
        获取数据集中的一个样本

        Get a sample from the dataset.

        参数/Parameters
        ----------
        idx : int
            样本的索引位置
            Index position of the sample

        返回/Returns
        -------
        item : dict
            包含图像、目标、额外信息、外参和内参的数据字典
            A dictionary containing images, targets, extra information, extrinsic and intrinsic data
        """

        token = self.tokens[idx]
        sample = np.load(os.path.join(self.anno_root, f'{token}.npz'), allow_pickle=True)
        # images
        images = []
        for im_view in self.img_key_list:
            for im_path in sample['image_paths']:
                if im_path.startswith(f'samples/{im_view}/'):
                    im_path = os.path.join(self.nusc_root, im_path)
                    img = np.asarray(Image.open(im_path))
                    images.append(img)
        # pivot pts
        pivot_pts = sample["pivot_pts"].item()
        valid_length = sample["pivot_length"].item()
        # targets
        masks=sample[self.mask_key]
        targets = dict(masks=masks, points=pivot_pts, valid_len=valid_length) 
        # pose
        extrinsic = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        extrinsic[:, :3, :3] = sample["rots"]
        extrinsic[:, :3, 3] = sample["trans"]
        intrinsic = sample['intrins']
        # transform
        item = dict(images=images, targets=targets,
                    extra_infos=dict(token=token, map_size=self.ego_size),
                    extrinsic=np.stack(extrinsic, axis=0), intrinsic=np.stack(intrinsic, axis=0))
        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.tokens)