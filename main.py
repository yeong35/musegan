# utils
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn

from data.utils import postProcess

# data load library
from torch.utils.data import DataLoader
from data.utils import MidiDataset

# model
from model import TempNetwork, BarGenerator, MuseCritic, MuseGenerator
from model import initialize_weights

# trainer
from trainer import Trainer

# ======= MAIN ===========
def train():
    # device setting
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset load
    dataset = MidiDataset(path='./data/chorales/Jsb16thSeparated.npz')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    # Model
    # tempNetwork
    tempnet = TempNetwork()
    x = torch.rand(10, 32)
    # BarGenerator
    bargenerator = BarGenerator()
    a = torch.rand(10, 128)

    # MuseGenerator
    generator = MuseGenerator()

    cords = torch.rand(10, 32)
    style = torch.rand(10, 32)
    melody = torch.rand(10, 4, 32)
    groove = torch.rand(10, 4, 32)

    # MuseGritic
    critic = MuseCritic()
    a = torch.rand(10, 4, 2, 16, 84)

    # Generator
    generator = MuseGenerator(n_tracks=4, n_bars=2, n_steps_per_bar=16, n_pitches=84,
                              z_dim=32, hid_channels=1024, hid_features=1024, out_channels=1).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.9))

    #Critic
    critic = MuseCritic(input_shape=(4, 2, 16, 84),
                    hid_channels=128,
                    hid_features=1024,
                    out_features=1).to(device)
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001, betas=(0.5, 0.9))

    generator = generator.apply(initialize_weights)
    critic = critic.apply(initialize_weights)

    # Training
    trainer = Trainer(generator, critic, g_optimizer, c_optimizer)
    trainer.train(dataloader, epochs=1000)

    losses = trainer.data.copy()

    # save loss
    df = pd.DataFrame.from_dict(losses)

    # save models
    generator = generator.eval().cpu()
    critic = critic.eval().cpu()
    torch.save(generator, 'generator_e1000.pt')
    torch.save(critic, 'critic_e1000.pt')

    #plot losses
    # plt.figure(figsize=(12, 8))
    # plt.plot(losses['gloss'][:500], 'orange', label='generator')
    # plt.plot(losses['cfloss'][:500], 'r', label='critic fake')
    # plt.plot(losses['crloss'][:500], 'g', label='critic real')
    # plt.plot(losses['cploss'][:500], 'b', label='critic penalty')
    # plt.plot(losses['closs'][:500], 'm', label='critic')
    # plt.xlabel('epoch', fontsize=12)
    # plt.ylabel('loss', fontsize=12)
    # plt.grid()
    # plt.legend()
    # plt.show()
    #plt.savefig('losses.png')

    generator = generator.eval().cpu()

    # make prediction
    chords = torch.rand(1, 32)
    style = torch.rand(1, 32)
    melody = torch.rand(1, 4, 32)
    groove = torch.rand(1, 4, 32)

    preds = generator(chords, style, melody, groove).detach()

    # Get music data 이건 뭐하는거지..?
    preds = preds.numpy()
    music_data = postProcess(preds)

    # save file
    filename = 'myexample.midi'
    music_data.write('midi', fp=filename)

if __name__=="__main__":
    train()