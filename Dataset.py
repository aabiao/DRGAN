import numpy as np 
import glob
import cv2
from torch.utils.data import Dataset
import pickle

def normalize_image(img):
    img = img - img.min()
    img = img / img.max()
    img = img - 1
    return img

class mydataset(Dataset):
    def __init__(self, path, test=False):
        f = open('ModelDict.pkl', 'rb')
        info = pickle.load(f)
        print("Processing images ...")
        print("Processing a new class of image")
        print("Class No: ", 0)
        image_paths = glob.glob(path)
        last_id = int(image_paths[0].split('/')[-2])
        new_id = 0
        images = []
        one_hot_labels = []
        pose = []
        for image in image_paths:
            if test:
                limit = 9
            else:
                limit = 1180
            pic = cv2.imread(image)
            pic = normalize_image(pic)
            tmp = cv2.resize(pic, (96, 96), interpolation=cv2.INTER_CUBIC)
            
            last, name = int(image.split('/')[-2]), image.split('/')[-1][:-4]
            if last == last_id:
                one_hot_labels.append(new_id)
                pose.append(info[name][1])
                images.append(tmp)

            else:
                if test and new_id >= limit:
                    break
                print("Processing a new class of image")
                print("Class No: ", new_id+1)
                new_id += 1
                last_id = last
                one_hot_labels.append(new_id)
                pose.append(info[name][1])
                images.append(tmp)
        self.images = np.array(images)
        self.IDs = np.array(one_hot_labels)
        self.poses = np.array(pose)
        print("Image: " , len(self.images))
        print("IDs: " , len(self.IDs))
        print("poses: " , len(self.poses))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        ID = self.IDs[idx]
        pose = self.poses[idx]

        return [image, ID, pose]

if __name__ == "__main__":
    result = mydataset("./cropcars_model/train/*/*.jpg", test=False)
    print(len(result))

