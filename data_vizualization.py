import streamlit as st
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

X_feat = np.load("features_np.npy")
Y_labels = np.load("labels.npy")


st.title('3D Scatter Plot')
tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X_feat)

data = {'X0': X_tsne[:,0], 'X1': X_tsne[:,1], 'X2': X_tsne[:,2], 'Y_labels': Y_labels}
df = pd.DataFrame(data)
fig = px.scatter_3d(df, x='X0', y='X1', z='X2',
              color='Y_labels')

st.plotly_chart(fig)