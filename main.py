from src.dataset import Dataset
from src.modello import Modello
import numpy as np
import pandas as pd

dataset = Dataset(data=None, columns=None)
'''
dataset.set_dataset('hf://datasets/lvwerra/red-wine/winequality-red.csv')

dataset.export_to_csv('red_wine_quality.csv')

print(dataset.data.head())

print(dataset.data.info())  #1559 osservations, 12 columns

print(dataset.data.describe())  

dataset.correlation_heatmap(top_n=12)
'''
dataset.data = pd.read_csv('red_wine_quality.csv')

model = Modello(dataset)

model.drop_colonna('quality')
model.split_data(0.2, 30)
model.set_modello(4, [256, 128, 64, 32], [0.3, 0.2, 0.1, 0])

model.early_stopping('val_loss', 100, True, 2)
model.train_model(1000, 32)
model.evaluate_model()

model.plot_confusion_matrix()