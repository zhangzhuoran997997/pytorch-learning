# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import torch
import numpy as np
from PIL import Image
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "D:/biancheng/python/python project/PennFudanPed/PennFudanPed/PedMasks/FudanPed00009_mask.png"
    mask = Image.open(path)
    mask = np.array(mask)
    print(mask[137,381])
    obj_ids = np.unique(mask)
    print(obj_ids)
    obj_ids = obj_ids[1:]
    masks = mask == obj_ids[:, None, None]
    print(masks.shape)
    num_objs = len(obj_ids)
    print(num_objs)
    labels = torch.ones((2), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    print(masks,masks.shape)
    tensor1 = torch.tensor(8)
    tensor2 = torch.tensor([1,2])
    print(tensor1 , tensor2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
