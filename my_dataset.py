from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms


class MyDataSet(Dataset):
    """"""

    def __init__(self, support_data: list, query_data: list, transform=None):
        self.support_data = support_data
        self.query_data = query_data
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.support_data)

    def __getitem__(self, item):
        s_img = Image.open(self.support_data[item][0]).convert('RGB')

        s_mask = Image.open(self.support_data[item][1]).convert('L')

        q_img = Image.open(self.query_data[item][0]).convert('RGB')

        q_mask = Image.open(self.query_data[item][1]).convert('L')

        if self.transform is not None:
            s_img_t = self.transform(s_img)
            q_img_t = self.transform(q_img)

        s_mask_t = self.to_tensor(s_mask)
        s_mask_t = self.resize(s_mask_t)
        q_mask_t = self.to_tensor(q_mask)
        q_mask_t = self.resize(q_mask_t)

        return s_img_t, s_mask_t, q_img_t, q_mask_t


class MyDataSet2(Dataset):
    """"""

    def __init__(self, support_data: list, query_data: list, transform=None):
        self.support_data = support_data
        self.query_data = query_data
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((257, 257))

    def __len__(self):
        return len(self.support_data)

    def __getitem__(self, item):
        s_img = Image.open(self.support_data[item][0]).convert('RGB')

        s_mask = Image.open(self.support_data[item][1]).convert('L')

        q_img = Image.open(self.query_data[item][0]).convert('RGB')

        q_mask = Image.open(self.query_data[item][1]).convert('L')

        if self.transform is not None:
            s_img_t = self.transform(s_img)
            q_img_t = self.transform(q_img)

        s_mask_t = self.to_tensor(s_mask)
        s_mask_t = self.resize(s_mask_t)
        q_mask_t = self.to_tensor(q_mask)
        q_mask_t = self.resize(q_mask_t)

        return s_img_t, s_mask_t, q_img_t, q_mask_t


class MyDataSet3(Dataset):
    """"""

    def __init__(self, support_img_l: list, support_mask_l: list, query_img_l: list, transform=None):
        self.support_img_l = support_img_l
        self.support_mask_l = support_mask_l
        self.query_img_l = query_img_l
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((257, 257))

    def __len__(self):
        return len(self.query_img_l)

    def __getitem__(self, item):
        s_img = Image.open(self.support_img_l[item]).convert('RGB')

        s_mask = Image.open(self.support_mask_l[item]).convert('L')

        q_img = Image.open(self.query_img_l[item]).convert('RGB')


        if self.transform is not None:
            s_img_t = self.transform(s_img)
            q_img_t = self.transform(q_img)

        s_mask_t = self.to_tensor(s_mask)
        s_mask_t = self.resize(s_mask_t)

        return s_img_t, s_mask_t, q_img_t
