# Define the dataset
import os
import numpy as np
import torch
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset):
    def __inf__(self, root, transforms):
        # set the root path of the dataset
        self.root = root
        # transfer the Data to tensor
        self.transforms = transforms
        # functions
        # os.path.join(path1,path2..) -> put the paths together
        # os.listdir(path) -> Return a list containing the names of the files in the directory.
        # sorted(list) -> sort the things in the list
        # load all image files and sort them
        self.imgs = list(sorted(os.listdir(os.path.join(root,"PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # function
        # open(path) -> load a file
        # convert() -> change the format of the image
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        # get the instances in the mask: 0->background 1->instance1 2->instance2
        # np.unique() -> 去除掉相同元素然后排序输出
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        # mask 与 每一个物体编号进行比较，eg1，2...，得到一个01的矩阵，代表着某个点是否是物体k
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            # function
            # np.where(array) -> output the coordinates where the elements
            # in the array is not 0
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # function
            # list.append(obj) -> add the obj to the list tail
            # eg. list = [123,'zzr'] list.append('wlq') -> list = [123,'zzr','wlq']
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # function torch.ones(size,dtype) -> a tensor shaped size with all one
        # size can be (2,3,4)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
