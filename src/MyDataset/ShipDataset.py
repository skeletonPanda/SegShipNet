from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import PIL.Image as Image


class ShipDataset(Dataset):
    '''
    船的分割数据集
    路径为labelme生成的文件夹
    '''
    def __init__(self, root, transform_img=transforms.ToTensor(), transform_label=transforms.ToTensor()):
        if not os.path.exists(root):
            raise 'Dictionary Not Found'

        data_path = []

        folders = os.listdir(root)
        for folder in folders:
            dir_path = os.path.join(root, folder)
            if os.path.isdir(dir_path):
                img_path = os.path.join(dir_path, 'img.png') # 合成路径
                label_path = os.path.join(dir_path, 'label.png')
                data_path.append((img_path, label_path))

        self.data_path = data_path
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __getitem__(self, item):
        img_path, label_path = self.data_path[item]
        img = Image.open(img_path)
        label = Image.open(label_path)

        if self.transform_img is not None:
            img = self.transform_img(img)
            label = self.transform_label(label)

        return img, label

    def __len__(self):
        return len(self.data_path)


if __name__ == '__main__':
    dataset = ShipDataset('../data/seg_ship_data/4-yyc_pic_output')
    print('length: ', len(dataset))
    # dataset[0][0].show()
    # dataset[0][1].show()
    dataloader = DataLoader(dataset, batch_size=64)
    batch_imgs, batch_labels = iter(dataloader).next()
    print(batch_imgs.size())
    print(batch_labels.size())