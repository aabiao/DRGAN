import torch
from torch.autograd import Variable
import os
import numpy as np 
import cv2
from model import Generator, Discriminator
from Dataset import normalize_image
import time
import train_single

def test_one(path):
	if not os.path.exists(path.split('/')[-2]):
		os.mkdir(path.split('/')[-2])
	class_label = path.split('/')[-2]
	print("Testing Class: ", class_label)
	print('Loading parameters...')
	G = Generator(3, 5, 50)
	D = Discriminator(1181, 5, 3)
	G.load_state_dict(torch.load('./checkpoints/net_params_G_100.pkl'))
	G.cuda()
	D.load_state_dict(torch.load('./checkpoints/net_params_D_100.pkl'))
	D.cuda()
	img = cv2.imread(path)
	img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
	img = normalize_image(img)
	images = np.array([img[:] for _ in range(5)])
	pose = np.array([0, 1, 2, 3, 4])
	noise = torch.FloatTensor(np.random.uniform(-1, 1, (5, 50)))
	print(noise)
	print(img)
	images = torch.FloatTensor(images)
	pose = torch.LongTensor(pose)
	pose_one_hot =  train_single.one_hot(pose, 5)
	noise = torch.FloatTensor(noise)
	images = Variable(images.type('torch.FloatTensor')).cuda().permute(0, 3, 1, 2)
	pose = Variable(pose.type('torch.FloatTensor')).cuda()
	noise = Variable(noise.type('torch.FloatTensor')).cuda()
	pose_one_hot = Variable(pose_one_hot.type('torch.FloatTensor')).cuda()
	print('Testing...')
	image_syn = G(images, pose_one_hot, noise)
	images = list(image_syn.permute(0, 2, 3, 1).data.cpu().numpy())
	for idx in range(5):
		image = np.array(images[idx])
		image = train_single.normalize_image(image)
		cv2.imwrite('./'+class_label+'/'+path.split('/')[-1][:-4]+'_'+str(idx)+'.jpg', image)
	D_output = D(image_syn)
	print(D_output[:, :1182])
	print(D_output[0, :1182].max())
	print(D_output[1, :1182].max())
	print(D_output[2, :1182].max())
	print(D_output[3, :1182].max())
	print(D_output[4, :1182].max())
	print('Test finished!')

if __name__ == '__main__':
	test_one("/home/qijiayi/Desktop/DRGAN/cropcars_model/test/1603/1c6da8d8e1ec84.jpg")