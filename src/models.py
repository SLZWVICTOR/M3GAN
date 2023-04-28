import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
torch.manual_seed(1)
import random
import os

class myGAN(nn.Module):
	def __init__(self, feats):
		super(myGAN, self).__init__()
		self.name = 'myGAN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_window = 5
		self.n_mask = int(self.n_window * 0.4) 
		self.n = self.n_feats * self.n_mask
		self.full_n = self.n_feats * self.n_window
		self.rth = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
		)
		self.gen = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.full_n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def pinjie(self, t1, t2, o1, o2):
		t1 = t1.view(-1, 1)
		t2 = t2.view(-1, 1)
		length1 = t1.size()[0]
		length2 = t2.size()[0]
		length = length1 + length2

		if 0 in o1:
			t = t1[o1.index(0)]
		elif 0 in o2:
			t = t2[o2.index(0)]

		for i in range(1, length):
			if i in o1:
				t = torch.cat((t, t1[o1.index(i)]), 0)
			elif i in o2:
				t = torch.cat((t, t2[o2.index(i)]), 0)
		return t

	def forward(self, g):
		LEN_PER_PART = 1
		len_after_mask = self.n_mask
		remain_list = []
		PART = self.n_window / LEN_PER_PART
		select_list = random.sample(range(0, int(PART)), int(len_after_mask / LEN_PER_PART))
		select_list = np.sort(select_list)
		temp_list = []
		for ii in range(len(select_list)):
			for jj in range(LEN_PER_PART):
				temp_list.append(LEN_PER_PART * select_list[ii] + jj)
		select_list = temp_list
		selected_list = select_list

		for v in range(self.n_window):
			if v not in selected_list:
				remain_list.append(v)

		selected_datalist = []
		remain_datalist = []
		for i in range(self.n_window):
			if i in selected_list:
				for j in range(self.n_feats):
					selected_datalist.append(j + i * self.n_feats)
			elif i in remain_list:
				for j in range(self.n_feats):
					remain_datalist.append(j + i * self.n_feats)

		ori_data = g.view(1, -1)
		selected_data = ori_data[:, selected_datalist]
		remain_data = ori_data[:, remain_datalist]

		z = self.gen(self.rth(selected_data))
		z2 = torch.cat([z, remain_data], dim=1)
		g2 = torch.cat([selected_data, remain_data], dim=1)

		real_score = self.discriminator(g2.view(1, -1))
		fake_score = self.discriminator(z2.view(1, -1))

		z_r = z2.view(-1)
		real_r = real_score.view(-1)
		fake_r = fake_score.view(-1)
		return z_r, real_r, fake_r


