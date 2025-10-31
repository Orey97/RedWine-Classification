from src.dataset import Dataset

dataset = Dataset(data=None, columns=None)
dataset.set_dataset('hf://datasets/lvwerra/red-wine/winequality-red.csv')

dataset.export_to_csv('red_wine_quality.csv')

print(dataset.data.head())

print(dataset.data.info())  #1559 osservations, 12 columns

print(dataset.data.describe())  

dataset.correlation_heatmap(top_n=12)