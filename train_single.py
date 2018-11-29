import numpy as np 
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Discriminator, Generator
from Dataset import mydataset
import time
import os
import cv2


def one_hot(labels, batch):
    ones = torch.sparse.torch.eye(batch)
    return ones.index_select(0, labels)

def normalize_image(img):
    img = img - img.min()
    img = img / img.max() * 256
    return img


def train(BATCHSIZE, MAX_EPOCH, ND, NP, NZ, IN_CHANNEL, LR, GPU, TRAINING_SET, TEST, dataset):
    D_model = Discriminator(ND, NP, IN_CHANNEL)
    G_model = Generator(IN_CHANNEL, NP, NZ)
    if GPU:
        D_model = D_model.cuda()
        G_model = G_model.cuda()
    optimizer_D = optim.Adam(D_model.parameters(), lr = LR)
    optimizer_G = optim.Adam(G_model.parameters(), lr = LR)
    for epoch in range(1, MAX_EPOCH + 1):
        step = 0
        dataloader = DataLoader(dataset, batch_size = BATCHSIZE, shuffle = True)
        sum1 = 0
        for _, batch_data in enumerate(dataloader):
            start = time.time()
            step += 1
            D_model.zero_grad()
            G_model.zero_grad()

            batch_image = batch_data[0]
            batch_id_label = batch_data[1] # one_hot_id_label length = ND
            batch_pose_label = batch_data[2] # one_hot_pose_label length = NP
            minibatch_size = len(batch_image) # current batch size

            pose = torch.LongTensor(np.random.randint(NP, size=minibatch_size))
            pose_one_hot = one_hot(pose, NP)
            noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, NZ)))
            batch_fake_label = torch.ones(minibatch_size) * ND
            batch_zeros_label = torch.ones(minibatch_size)
            # transform variables
            
            batch_image, batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label = \
            Variable(batch_image.type('torch.FloatTensor')), Variable(batch_id_label.type('torch.LongTensor')), Variable(batch_pose_label.type('torch.LongTensor')), Variable(batch_fake_label.type('torch.LongTensor')), Variable(batch_zeros_label.type('torch.FloatTensor'))

            batch_image = batch_image.permute(0, 3, 1, 2)
            noise, pose_one_hot, pose = Variable(noise.type('torch.FloatTensor')), Variable(pose_one_hot.type('torch.FloatTensor')), Variable(pose)           
            if GPU:
                batch_image, batch_id_label, batch_pose_label, batch_fake_label, batch_zeros_label, pose = \
                batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda(), pose.cuda()

                noise, pose_one_hot, pose = noise.cuda(), pose_one_hot.cuda(), pose.cuda()

            image_syn = G_model(batch_image, pose_one_hot, noise)
            loss_crossentropy = nn.CrossEntropyLoss()
            loss_BCElog = nn.BCEWithLogitsLoss()
            if GPU:
                loss_BCElog = loss_BCElog.cuda()
                loss_crossentropy = loss_crossentropy.cuda()
            # train D
            real_output = D_model(batch_image)
            syn_output = D_model(image_syn.detach())

            loss_id_D = loss_crossentropy(real_output[:, :ND+1], batch_id_label) + loss_crossentropy(syn_output[:, :ND+1], batch_fake_label)
            # loss_GAN_D = loss_BCElog(real_output[:, ND], batch_ones_label) + loss_BCElog(syn_output[:, ND], batch_zeros_label)
            loss_pose_D = loss_crossentropy(real_output[:, ND+1:], batch_pose_label)
            loss_D = loss_id_D  + 5*loss_pose_D

            loss_D.backward()
            optimizer_D.step()

            # train G
            syn_output = D_model(image_syn)

            loss_id_G = loss_crossentropy(syn_output[:, :ND+1], batch_id_label)

            # mark 11/25
            # loss_GAN_G = loss_BCElog(syn_output[:, ND], batch_ones_label) 
            loss_pose_G = loss_crossentropy(syn_output[:, ND+1:], pose)

            loss_G = loss_id_G + 5*loss_pose_G

            loss_G.backward()
            optimizer_G.step()
            stop = time.time()
            sum1 += stop-start
            if step % 5 == 0:
                text = "EPOCH : {0}, step : {1}, {2} : {3}, {4} : {5}".format(epoch, step, 'D', loss_D.data.cpu().numpy()[0], 'G', loss_G.data.cpu().numpy()[0])
                print(text)
                print("Last 5 steps time: ", sum1, " s")
                sum1 = 0
            
                # save images
                images = list(image_syn.permute(0, 2, 3, 1).data.cpu().numpy())
                for idx in range(minibatch_size):
                    image = np.array(images[idx])
                    image = normalize_image(image)
                    cv2.imwrite('./generated_images/result_'+str(idx)+'.jpg', image)

        # save parameters
        torch.save(D_model.state_dict(), './checkpoints/net_params_D_' + str(epoch) + '.pkl')
        torch.save(G_model.state_dict(), './checkpoints/net_params_G_' + str(epoch) + '.pkl')

        # max to keep 4 checkpoints
        if epoch >= 5:
            os.remove('./checkpoints/net_params_D_' + str(epoch-4) + '.pkl')
            os.remove('./checkpoints/net_params_G_' + str(epoch-4) + '.pkl')

        '''
        def restore_params():
            # 新建 net3
            net3 = torch.nn.Sequential(
                torch.nn.Linear(1, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 1)
            )

            # 将保存的参数复制到 net3
            net3.load_state_dict(torch.load('net_params.pkl'))
            prediction = net3(x)
        '''

if __name__ == "__main__":
    BATCHSIZE = 96
    MAX_EPOCH = 100
    NZ = 50
    IN_CHANNEL = 3
    LR = 0.0002
    GPU = 1
    TRAINING_SET = './cropcars_model/train/*/*.jpg'
    TEST = True
    dataset = mydataset(TRAINING_SET, TEST)
    ND = len(set(tuple(dataset.IDs)))
    NP = len(set(tuple(dataset.poses)))
    print(ND)
    print(NP)
    train(BATCHSIZE, MAX_EPOCH, ND, NP, NZ, IN_CHANNEL, LR, GPU, TRAINING_SET, TEST, dataset)