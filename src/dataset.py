import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras


class Dataset:
    def __init__(self, data, columns): 
        self.data = data
        self.columns = columns

    def set_dataset(self, url):
        self.data = pd.read_csv(url)
        self.columns = self.data.columns.tolist()

    def export_to_csv(self, file_path):
        self.data.to_csv(file_path, index=False)


    def correlation_heatmap(self, top_n=20):
        """Mostra una heatmap delle correlazioni tra le variabili numeriche più importanti."""
        numeric_df = self.data.select_dtypes(include=['number'])
        if numeric_df.empty:
            print("Nessuna colonna numerica trovata.")
            return

        corr = numeric_df.corr()
        top_corr = corr.nlargest(top_n, 'SalePrice')['SalePrice'].index \
                    if 'SalePrice' in corr.columns else corr.columns[:top_n]
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr.loc[top_corr, top_corr], annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Heatmap di correlazione (top {top_n})")
        plt.show()



    
        
