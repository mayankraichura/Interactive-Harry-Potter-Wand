import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

data = pd.read_csv('A_ZHandwrittenData.csv')
data[data.columns[0]].value_counts().sort_index(ascending=True)
for i in range(26):
    dd = data[data['0']==i].iloc[1]
    x = dd[1:].values
    x = x.reshape((28, 28))
    im = plt.subplot(5, 6, i+1)
    im.imshow(x, cmap='gray')