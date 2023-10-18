import numpy as np
from neural_network.ResNet import ResNet
from games.ConnectFour import ConnectFour
from games.Tictactoe import TicTacToe
from games.Gomoku import Gomoku
from AlphaCaro.AlphaCaroParallel import AlphaCaroParallel
import torch
import matplotlib.pyplot as plt

game = Gomoku()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 6, 64, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaCaroParallel(model, optimizer, game, args)
alphaZero.learn()