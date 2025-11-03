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

classification = Modello(dataset)

classification.drop_colonna('quality')
classification.split_data_less_classes(0.2, 420)
classification.set_modello(4, [256, 128, 64, 32], [0.2, 0.2, 0.1, 0])

classification.early_stopping('val_loss', 100, True, 2)
classification.train_model(1000, 32)
classification.evaluate_model()

classification.plot_confusion_matrix()

classification.model.save("classification_model.keras")