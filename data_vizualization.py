import streamlit as st
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
# import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as img


X_feat = np.load("features_np.npy")
Y_labels = np.load("labels.npy")


st.title('3D Scatter Plot')
tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X_feat)

data = {'X0': X_tsne[:,0], 'X1': X_tsne[:,1], 'X2': X_tsne[:,2], 'Y_labels': Y_labels}
df = pd.DataFrame(data)
# fig = px.scatter_3d(df, x='X0', y='X1', z='X2',
#               color='Y_labels')

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each class
for cls, color in zip(df['Y_labels'].unique(), ['r', 'g', 'b', 'y']):
    ix = df['Y_labels'] == cls
    ax.scatter(df.loc[ix, 'X0'], df.loc[ix, 'X1'], df.loc[ix, 'X2'], c=color, label='Y_labels {}'.format(cls))

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
# ax.set_title('3D Scatter Plot of Different Classes')
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
fig.savefig("3d_plot.png")

# Display the figure using st.pyplot()
st.pyplot(fig)


st.title('3D Scatter Plot')
# Display the image
image_path = 'classification_report.jpg'  
st.image(image_path, caption='Your Image', use_column_width=True)