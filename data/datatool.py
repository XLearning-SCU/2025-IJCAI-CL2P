
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os
from torchvision.utils import make_grid
import cv2
import torch.nn.functional as F
import glob
from torchvision import datasets
import json
import pandas as pd
# provent the depandency of multiple threads.
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


# -------------------------------------------------------------------------------
#                      Functional area.
# -------------------------------------------------------------------------------


def plain_transforms(img):
    """
    plain transformation operation.
    """
    return img


def coil(root, n_objs=20, n_views=3):
    """
    Download: 
    https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php

    1. coil-20: 
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip


    2. coil-100:
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
    """
    
    if os.path.isfile(os.path.join(root, f"coil-{n_objs}/{n_views}v-cache.pth")):
        print('load cache')
        X_train, X_test, y_train, y_test = torch.load(os.path.join(root, f"coil-{n_objs}/{n_views}v-cache.pth"))
        return X_train, X_test, y_train, y_test
    else:
        from skimage.io import imread
        from sklearn.model_selection import train_test_split
        assert n_objs in [20, 100]
        data_dir = os.path.join(root, f"coil-{n_objs}")
        img_size = (1, 128, 128) if n_objs == 20 else (3, 128, 128)
        n_imgs = 72

        n = (n_objs * n_imgs) // n_views

        views = []
        labels = []

        img_idx = np.arange(n_imgs)

        # rng = np.random.default_rng(seed=42)
        for obj in range(n_objs):
            obj_list = []
            # 让每次的数据集多视图划分一致
            # obj_img_idx = rng.permutation(img_idx).reshape(n_views, n_imgs // n_views)            
            
            obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
            
            labels += (n_imgs // n_views) * [obj]

            for view, indices in enumerate(obj_img_idx):
                sub_view = []
                for i, idx in enumerate(indices):
                    if n_objs == 20:
                        fname = os.path.join(data_dir, f"obj{obj + 1}__{idx}.png")
                        img = imread(fname)[None, ...]
                    else:
                        fname = os.path.join(
                            data_dir, f"obj{obj + 1}__{idx * 5}.png")
                        img = imread(fname)
                    if n_objs == 100:
                        img = np.transpose(img, (2, 0, 1))
                    sub_view.append(img)
                obj_list.append(np.array(sub_view))
            views.append(np.array(obj_list))
        views = np.array(views)
        views = np.transpose(views, (1, 0, 2, 3, 4, 5)
                            ).reshape(n_views, n, *img_size)
        cp = views.reshape(-1, *img_size)
        # print(cp.shape)
        # print(cp[:, 0, :].mean(), cp[:, 0, :].std())
        labels = np.array(labels)
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            list(range(n)), labels, test_size=0.2, random_state=42)
        X_train, X_test = views[:, X_train_idx, :, :, :], views[:, X_test_idx, :, :, :] 
        # torch.save((X_train, X_test, y_train, y_test), os.path.join(root, f"coil-{n_objs}/{n_views}v-cache.pth"))
        # print('save cahce.')
        return X_train, X_test, y_train, y_test


def get_transformation(args):
    return transforms.Compose([
        transforms.Resize((args.dataset.crop_size, args.dataset.crop_size)),
        transforms.ToTensor()])


def edge_transformation(img):
    """
    edge preprocess functuin.
    """
    trans = transforms.Compose([image_edge, transforms.ToPILImage()])
    return trans(img)


def image_edge(img):
    """
    :param img:
    :return:
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return edge


def generate_tiny_dataset(name, dataset, sample_num=100):
    """
    Tiny data set for T-SNE to visualize the representation's structure.
    Only support Edge-MNIST, Edge-Fashion.
    """
    assert name in ['Edge-MNIST', 'Edge-Fashion']
    y = dataset.targets.unique()
    x1s = []
    x2s = []
    ys = []

    for _ in y:
        idx = dataset.targets == _
        x1, x2, yy = dataset.data[0][idx, :], dataset.data[1][idx, :], dataset.targets[idx]
        x1, x2, yy = x1[:sample_num], x2[:sample_num], yy[:sample_num]
        x1s.append(x1)
        x2s.append(x2)
        ys.append(yy)

    x1s = torch.vstack(x1s)
    x2s = torch.vstack(x2s)
    ys = torch.concat(ys)

    tiny_dataset = {
        "x1": x1s,
        "x2": x2s,
        "y": ys
    }
    os.makedirs("./experiments/tiny-data/", exist_ok=True)
    torch.save(tiny_dataset, f'./experiments/tiny-data/{name}_tiny.plk')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def align_office31(root):
    """
    This function will auto fill the lack data by color jitter method.
    The original Office dataset has 4,110 items. After alignment, it has 8,451 items
    """
    from tqdm import tqdm
    from glob import glob
    from torchvision.utils import save_image
    from sklearn.model_selection import train_test_split
    import json
    
    def padding_images(transform, image_dir, max_num):
        if len(image_dir) == max_num:
            return None
        index = torch.arange(len(image_dir))
        repeat_time = (max_num // len(image_dir))+1
        index = index.repeat(repeat_time)

        for n, idx in enumerate(index[:max_num-len(image_dir)]):
            img_path = image_dir[idx]
            new_path = f"{img_path[:-4]}_jitter_{n}.jpg"
            image = pil_loader(img_path)
            t_image = transform(image)
            save_image(t_image, new_path)

    views_mapping = {
        'A': 'amazon/images',
        'D': 'dslr/images',
        'W': 'webcam/images'
    }

    classes = ['paper_notebook', 'desktop_computer', 'punchers', 'desk_lamp', 'tape_dispenser',
               'projector', 'calculator', 'file_cabinet', 'back_pack', 'stapler', 'ring_binder',
               'trash_can', 'printer', 'bike', 'mug', 'scissors', 'bike_helmet', 'mouse', 'bookcase',
               'pen', 'bottle', 'keyboard', 'phone', 'ruler', 'headphones', 'speaker', 'letter_tray',
               'monitor', 'mobile_phone', 'desk_chair', 'laptop_computer']

    jitter = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ColorJitter(brightness=.5, hue=.3,
                                   contrast=.3, saturation=.3),
            transforms.ToTensor()
        ]
    )
    
    for c in tqdm(classes):
        
        item_A = glob(f"{root}/{views_mapping['A']}/{c}/*.jpg")
        item_D = glob(f"{root}/{views_mapping['D']}/{c}/*.jpg")
        item_W = glob(f"{root}/{views_mapping['W']}/{c}/*.jpg")
        max_num = max(len(item_A), len(item_D), len(item_W))

        padding_images(jitter, item_A, max_num)
        padding_images(jitter, item_D, max_num)
        padding_images(jitter, item_W, max_num)
    
    A_path = []
    D_path = []
    W_path = []
    targets = []
    # split into train set and test set.    
    for idx, c in enumerate(classes):
        item_A = glob(f"{root}/{views_mapping['A']}/{c}/*.jpg")
        item_D = glob(f"{root}/{views_mapping['D']}/{c}/*.jpg")
        item_W = glob(f"{root}/{views_mapping['W']}/{c}/*.jpg")
        
        A_path += [p[len(root):] for p in item_A]
        D_path += [p[len(root):] for p in item_D]
        W_path += [p[len(root):] for p in item_W]
        targets += ([idx] * len(item_A))
        print(A_path)
    X = np.c_[[A_path, D_path, W_path]].T
    Y = np.array(targets)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train = []
    for (a, d, w), y in zip(X_train, y_train):
        train.append((a, d, w, int(y)))
        
    test = []
    for (a, d, w), y in zip(X_test, y_test):
        test.append((a, d, w, int(y)))
        
    json.dump(train, open(f'{root}/train.json', 'w'))
    json.dump(test, open(f'{root}/test.json', 'w'))



class EdgeMNISTDataset(torchvision.datasets.MNIST):
    """
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]


class EdgeFashionDataset(torchvision.datasets.FashionMNIST):

    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]


class MultiCOIL20Dataset(Dataset):

    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(20))
        self.train = train
        self.views = views
        self.to_pil = transforms.ToPILImage()
        X_train, X_test, y_train, y_test = coil(
            self.root, n_objs=20, n_views=self.views)
        if self.train:
            self.data = X_train
            self.targets = torch.from_numpy(y_train).long()
        else:
            self.data = X_test
            self.targets = torch.from_numpy(y_test).long()

    def __getitem__(self, index):
        views = [np.transpose(self.data[view, index, :], (1, 2, 0))
                 for view in range(self.views)]
        target = self.targets[index]

        views = [self.to_pil(v) for v in views]

        if self.transform:
            views = [self.transform(x) for x in views]

        if self.target_transform:
            target = self.target_transform(target)

        return views, target

    def __len__(self) -> int:
        return self.data.shape[1]


class MultiCOIL100Dataset(Dataset):
    
    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(100))
        self.train = train
        self.views = views
        self.to_pil = transforms.ToPILImage()
        X_train, X_test, y_train, y_test = coil(
            self.root, n_objs=100, n_views=self.views)
        if self.train:
            self.data = X_train
            self.targets = torch.from_numpy(y_train).long()
        else:
            self.data = X_test
            self.targets = torch.from_numpy(y_test).long()

    def __getitem__(self, index):
        views = [np.transpose(self.data[view, index, :], (1, 2, 0))
                 for view in range(self.views)]
        target = self.targets[index]

        views = [self.to_pil(v) for v in views]

        if self.transform:
            views = [self.transform(x) for x in views]

        if self.target_transform:
            target = self.target_transform(target)
        
        return views, target

    def __len__(self) -> int:
        return self.data.shape[1]


class MultiOffice31Dataset(Dataset):
    """
    Before use our MultiOffice31Dataset, you should firstly run the `align_office31` function.
    After that, you will get the alignment dataset, train.json, and test.json files.
    Stats:
        Totoal number: (2817, 3) (2817,)
        Train set: (2253, 3) (2253, )
        Test set: (564, 3) (564, )
    """
    
    views_mapping = {
        'A': 'amazon/images',
        'D': 'dslr/images',
        'W': 'webcam/images'
    }

    classes = ['paper_notebook', 'desktop_computer', 'punchers', 'desk_lamp', 'tape_dispenser',
               'projector', 'calculator', 'file_cabinet', 'back_pack', 'stapler', 'ring_binder',
               'trash_can', 'printer', 'bike', 'mug', 'scissors', 'bike_helmet', 'mouse', 'bookcase',
               'pen', 'bottle', 'keyboard', 'phone', 'ruler', 'headphones', 'speaker', 'letter_tray',
               'monitor', 'mobile_phone', 'desk_chair', 'laptop_computer']

    def __init__(self, root='datasets/Office31', train: bool = True,
                 transform=None, target_transform=None, download: bool = False, views=3) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.load_image_path(train)
        
        
    def load_image_path(self, train):
        import json
        if train:
            self.data = json.load(open(os.path.join(self.root, 'train.json')))
        else:
            self.data = json.load(open(os.path.join(self.root, 'test.json')))
    
    def __getitem__(self, index):
        a, d, w, target = self.data[index]

        view0 = pil_loader(self.root+a)
        view1 = pil_loader(self.root+d)
        view2 = pil_loader(self.root+w)                
        # view0 = pil_loader(os.path.join(self.root, a))
        # view1 = pil_loader(os.path.join(self.root, d))
        # view2 = pil_loader(os.path.join(self.root, w))
        target = torch.tensor(target).long()

        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        return [view0, view1, view2], target
  
            
    def __len__(self):
        return len(self.data)


__dataset_dict = {
    'Edge-MNIST': EdgeMNISTDataset,
    'Edge-Fashion': EdgeFashionDataset,
    'Multi-COIL-20': MultiCOIL20Dataset,
    'Multi-COIL-100': MultiCOIL100Dataset,
    'Multi-Office-31': MultiOffice31Dataset,
}


def get_train_dataset(args, transform):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    train_set = data_class(root=args.dataset.root, train=True,
                            transform=transform, download=True, views=args.dataset.views)

    return train_set


def get_val_dataset(args, transform):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    val_set = data_class(root=args.dataset.root, train=False,
                        transform=transform, views=args.dataset.views)

    return val_set


if __name__ == '__main__':
    
    transformations = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()])
    dataset = MultiOffice31Dataset(root='data/raw/Office31', train=True,
                                    transform=transformations, views=3)