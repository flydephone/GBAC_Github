import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats # Import the stats module from SciPy
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import pickle
import utils

sns.set(style="whitegrid")
seed = 42
np.random.seed(seed)

# Load the datasets
# Note: Ensure the CSV files are in the same directory as this script.
print("Loading datasets, may take 60s - 100s...")
# Setting index_col=0 to correctly load the SNP names as the index
genotypes = pd.read_csv('Genotypes.csv', index_col="SNP_Name").T
environments = pd.read_csv('Environment_data.csv',index_col="ENV_Variable").T

genotypes.index.name = 'Hybrid'
environments.index.name = 'Environment'

cols_to_drop = [col for col in environments.columns if environments[col].nunique() == 1]
environments = environments.fillna(environments.mean(numeric_only=True))
environments = environments.drop(columns=cols_to_drop)

tra_phenotypes = pd.read_csv('Phenotypes.csv')
tes_phenotypes = pd.read_csv('Hybrids_to_be_predicted.csv')

tra_phenotypes.rename(columns={'Trait_1': 'Yield'}, inplace=True)
tra_phenotypes.rename(columns={'Trait_2': 'Moisture'}, inplace=True)
tes_phenotypes.rename(columns={'Trait_1': 'Yield'}, inplace=True)
tes_phenotypes.rename(columns={'Trait_2': 'Moisture'}, inplace=True)

genotypes_numeric = utils.encode_with_intermediate_map(genotypes)

# %% visualize the group of genotypes
pca = PCA(n_components=5)
df_pca = pd.DataFrame(pca.fit_transform(genotypes_numeric))

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
clusters = kmeans.fit_predict(df_pca)
df_pca['cluster'] = clusters

pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(genotypes_numeric)

vis_df = pd.DataFrame(X_pca2, columns=['PC1_vis', 'PC2_vis'])
vis_df['cluster'] = clusters

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))

# Scatter plot
scatter = sns.scatterplot(
    data=vis_df,
    x='PC1_vis',
    y='PC2_vis',
    hue='cluster',
    palette=sns.color_palette("husl", n_clusters),
    s=50,
    alpha=0.7,
    edgecolor='k',
    linewidth=0.5,
    ax=ax
)

# Customize the plot
ax.set_title(f'K-Means Clustering Results (Visualized in 2D with PCA)', fontsize=18, fontweight='bold')
ax.set_xlabel('Principal Component 1 (for Visualization)', fontsize=12)
ax.set_ylabel('Principal Component 2 (for Visualization)', fontsize=12)
ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
print("Displaying the plot. You may need to close the plot window to continue.")
plt.show()

# two traits
tra_traits_joint_plot = sns.jointplot(data=tra_phenotypes[['Moisture', 'Yield']], x='Moisture', y='Yield', kind='scatter', height=7)
tra_traits_joint_plot.fig.suptitle('Joint Plot of Moisture and Yield', y=1.02)
slope, intercept, r_value, p_value, std_err = stats.linregress(tra_phenotypes.dropna()['Moisture'], tra_phenotypes.dropna()['Yield'])
r_squared = r_value**2
tra_traits_joint_plot.ax_joint.text(
    0.05, 0.95, f'$R^2 = {r_squared:.2f}$', # The text to display
    transform=tra_traits_joint_plot.ax_joint.transAxes, # Use axis coordinates
    ha='left', # Horizontal alignment
    va='top', # Vertical alignment
    fontsize=12,
    bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5) # Optional text box
)
plt.show()

# %% GxE heatmap
# 1. Get all unique Hybrids and Environments from both dataframes
all_hybrids = pd.Index(tra_phenotypes['Hybrid'].unique()).union(tes_phenotypes['Hybrid'].unique())
all_environments = pd.Index(tra_phenotypes['Environment'].unique()).union(tes_phenotypes['Environment'].unique())

# 2. Create boolean masks indicating the presence of data for each file
# These masks are aligned to the full set of hybrids and environments
tra_locations = pd.crosstab(tra_phenotypes['Hybrid'], tra_phenotypes['Environment']).reindex(index=all_hybrids, columns=all_environments, fill_value=0) > 0
tes_locations = pd.crosstab(tes_phenotypes['Hybrid'], tes_phenotypes['Environment']).reindex(index=all_hybrids, columns=all_environments, fill_value=0) > 0

# 3. Create a single matrix for plotting.
plot_matrix = pd.DataFrame(np.nan, index=all_hybrids, columns=all_environments)


plot_matrix[tra_locations] = 0
plot_matrix[tes_locations] = 1

sum(plot_matrix.values)
# --- Plotting ---
fig, ax = plt.subplots(figsize=(22, 12))
cmap = ListedColormap(['blue', 'red'])
ax.grid(False)
sns.heatmap(plot_matrix,cmap=cmap,alpha=0.5,cbar=False,linewidths=0,ax=ax)

tra_patch = mpatches.Patch(color='blue',alpha=0.5, ec='black', label='Training')
tes_patch = mpatches.Patch(color='red',alpha=0.5, label='Test')
plt.legend(handles=[tra_patch, tes_patch], bbox_to_anchor=(0, 0), fontsize=12)

plt.title('Combined Genotype x Environment Matrix', fontsize=16)
plt.xlabel('Environment', fontsize=12)
plt.ylabel('Hybrid', fontsize=12)
plt.show()

# %% overlap of G and E in differernt dataset
# Get unique sets for hybrids and environments
tra_hybrids, tes_hybrids = set(tra_phenotypes['Hybrid']), set(tes_phenotypes['Hybrid'])
tra_envs, tes_envs = set(tra_phenotypes['Environment']), set(tes_phenotypes['Environment'])

hybrid_partitions = utils.get_overlap_partitions(('Train', tra_hybrids), ('Test', tes_hybrids))
env_partitions = utils.get_overlap_partitions(('Train', tra_envs), ('Test', tes_envs))

partition_order = ['Train-Only', 'Test-Only', 'Common (All)']

h_labels = [f"{'Common' if key == 'Common (All)' else key}\n({len(hybrid_partitions[key])})" for key in partition_order]
h_counts = [len(hybrid_partitions[key]) for key in partition_order]
h_pos = np.cumsum([0] + h_counts)

e_labels = [f"{'Common' if key == 'Common (All)' else key}\n({len(env_partitions[key])})" for key in partition_order]
e_counts = [len(env_partitions[key]) for key in partition_order]
e_pos = np.cumsum([0] + e_counts)

# --- Plotting the Mosaic ---
fig, ax = plt.subplots(figsize=(12, 10))
colors = {'train': 'blue', 'test': 'red'}

for i, h_key in enumerate(partition_order):
    for j, e_key in enumerate(partition_order):
        # Get the actual sets for the current partitions
        current_h_set = hybrid_partitions[h_key]
        current_e_set = env_partitions[e_key]

        # Check if the cell's components belong to the final datasets
        is_in_X_tra = current_h_set.issubset(tra_hybrids) and current_e_set.issubset(tra_envs)
        is_in_X_tes = current_h_set.issubset(tes_hybrids) and current_e_set.issubset(tes_envs)

        # Draw rectangles with alpha blending
        if is_in_X_tra:
            ax.add_patch(patches.Rectangle((e_pos[j], h_pos[i]), e_counts[j], h_counts[i], facecolor=colors['train'], alpha=0.4, linewidth=0))
        if is_in_X_tes:
            ax.add_patch(patches.Rectangle((e_pos[j], h_pos[i]), e_counts[j], h_counts[i], facecolor=colors['test'], alpha=0.4, linewidth=0))

ax.set_xlim(0, e_pos[-1])
ax.set_ylim(0, h_pos[-1])

ax.set_xticks(e_pos[:-1] + np.array(e_counts) / 2)
ax.set_xticklabels(e_labels, fontsize=12)
ax.set_yticks(h_pos[:-1] + np.array(h_counts) / 2)
ax.set_yticklabels(h_labels, fontsize=12)

ax.tick_params(axis='x', which='major', pad=10)
ax.tick_params(axis='y', which='major', pad=5)

ax.set_xlabel("Environments", fontsize=14)
ax.set_ylabel("Hybrids", fontsize=14)
ax.set_title("Mosaic Overlap of Hybrid and Environment Sets", fontsize=16)

# Create a custom legend
legend_patches = [
    patches.Patch(color=colors['train'], label=f"Training Set ({len(tra_hybrids)} H, {len(tra_envs)} E)", alpha=0.6),
    patches.Patch(color=colors['test'], label=f"Test Set ({len(tes_hybrids)} H, {len(tes_envs)} E)", alpha=0.6)
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=12)
plt.show()
# %% correlation of enviroment variables
correlation_matrix = environments.corr()

plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Environmental Variables')


# %% a demo to show splitting dataset and visualize
print("Splitting training data into Train and Validation sets...")
# Create the new dataframes based on the environment split
data_tra, data_val = utils.split_by_group(tra_phenotypes, split_col=('Environment','Hybrid'), train_ratio=0.7, seed=42)
data_tes = tes_phenotypes

tra_locations = pd.crosstab(data_tra['Hybrid'], data_tra['Environment']).reindex(index=all_hybrids, columns=all_environments, fill_value=0) > 0
val_locations = pd.crosstab(data_val['Hybrid'], data_val['Environment']).reindex(index=all_hybrids, columns=all_environments, fill_value=0) > 0
tes_locations = pd.crosstab(data_tes['Hybrid'], data_tes['Environment']).reindex(index=all_hybrids, columns=all_environments, fill_value=0) > 0

plot_matrix = pd.DataFrame(np.nan, index=all_hybrids, columns=all_environments)

plot_matrix[tra_locations] = 0
plot_matrix[val_locations] = 1
plot_matrix[tes_locations] = 2

sum(plot_matrix.values)
# --- Plotting ---
fig, ax = plt.subplots(figsize=(22, 12))
cmap = ListedColormap(['blue', 'green', 'red'])
ax.grid(False)
sns.heatmap(plot_matrix,cmap=cmap,alpha=0.5, cbar=False,linewidths=0,ax=ax)

tra_patch = mpatches.Patch(color='blue', alpha=0.5,ec='black', label='Training')
val_patch = mpatches.Patch(color='green', alpha=0.5,ec='black', label='Validation')
tes_patch = mpatches.Patch(color='red', alpha=0.5,label='Test')
plt.legend(handles=[tra_patch,val_patch, tes_patch], bbox_to_anchor=(0, 0), fontsize=12)

plt.title('Combined Genotype x Environment Matrix', fontsize=16)
plt.xlabel('Environment', fontsize=12)
plt.ylabel('Hybrid', fontsize=12)
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to prevent legend from being cut off
plt.show()

# --- 3. Visualization Option 2: Mosaic Overlap Plot ---
# --- Extract Unique Sets for Mosaic Plot ---
tra_hybrids, tra_envs = set(data_tra['Hybrid']), set(data_tra['Environment'])
val_hybrids, val_envs = set(data_val['Hybrid']), set(data_val['Environment'])
tes_hybrids, tes_envs = set(data_tes['Hybrid']), set(data_tes['Environment'])

# Calculate partitions for both hybrids and environments
hybrid_partitions = utils.get_overlap_partitions(('Train', tra_hybrids), ('Val', val_hybrids), ('Test', tes_hybrids))
env_partitions = utils.get_overlap_partitions(('Train', tra_envs), ('Val', val_envs), ('Test', tes_envs))

# Define a consistent order for plotting the partitions.
partition_order = [
    'Train-Only', 'Val-Only', 'Test-Only',
    'Common (T&V)', 'Common (T&s)', 'Common (V&s)',
    'Common (All)'
]

# Get counts and cumulative positions for drawing
# The labels replace 's' back with 't' for correct display on the plot
h_labels = [f"{key.replace('s', 't')}\n({len(hybrid_partitions[key])})" for key in partition_order]
h_counts = [len(hybrid_partitions[key]) for key in partition_order]
h_pos = np.cumsum([0] + h_counts)

e_labels = [f"{key.replace('s', 't')}\n({len(env_partitions[key])})" for key in partition_order]
e_counts = [len(env_partitions[key]) for key in partition_order]
e_pos = np.cumsum([0] + e_counts)

# --- Plotting the Mosaic ---
fig, ax = plt.subplots(figsize=(16, 12))

colors = {'train': 'blue', 'val': 'green', 'test': 'red'}

# Iterate through the 7x7 grid and draw colored patches
for i, h_key in enumerate(partition_order):
    for j, e_key in enumerate(partition_order):
        # Get the actual sets for the current partitions
        current_h_set = hybrid_partitions[h_key]
        current_e_set = env_partitions[e_key]

        # CORRECTED LOGIC:
        # Check if this combination of partitions exists in each of the final datasets
        # A cell is in a dataset if its hybrids AND its environments are subsets of that dataset's total hybrids/environments
        is_in_X_tra = current_h_set.issubset(tra_hybrids) and current_e_set.issubset(tra_envs)
        is_in_X_val = current_h_set.issubset(val_hybrids) and current_e_set.issubset(val_envs)
        is_in_X_tes = current_h_set.issubset(tes_hybrids) and current_e_set.issubset(tes_envs)

        # Draw rectangles with alpha blending. This will correctly layer the colors.
        if is_in_X_tra:
            ax.add_patch(patches.Rectangle((e_pos[j], h_pos[i]), e_counts[j], h_counts[i], facecolor=colors['train'], alpha=0.3, linewidth=0))
        if is_in_X_val:
            ax.add_patch(patches.Rectangle((e_pos[j], h_pos[i]), e_counts[j], h_counts[i], facecolor=colors['val'], alpha=0.3, linewidth=0))
        if is_in_X_tes:
            ax.add_patch(patches.Rectangle((e_pos[j], h_pos[i]), e_counts[j], h_counts[i], facecolor=colors['test'], alpha=0.3, linewidth=0))

# Add grid lines to separate the partitions
for pos in e_pos[1:-1]:
    ax.axvline(pos, color='grey', linestyle='-', linewidth=1, alpha=0.7)
for pos in h_pos[1:-1]:
    ax.axhline(pos, color='grey', linestyle='-', linewidth=1, alpha=0.7)

# --- Final plot adjustments ---
ax.set_xlim(0, e_pos[-1])
ax.set_ylim(0, h_pos[-1])

ax.set_xticks(e_pos[:-1] + np.array(e_counts) / 2)
ax.set_xticklabels(e_labels, fontsize=9)
ax.set_yticks(h_pos[:-1] + np.array(h_counts) / 2)
ax.set_yticklabels(h_labels, fontsize=9)

ax.tick_params(axis='x', which='major', pad=10)
ax.tick_params(axis='y', which='major', pad=5)


ax.set_xlabel("Environments", fontsize=14)
ax.set_ylabel("Hybrids", fontsize=14)
ax.set_title("Mosaic Overlap of Hybrid and Environment Sets", fontsize=16)

# Create a custom legend
legend_patches = [patches.Patch(color=colors['train'], label=f"Training Set ({len(tra_hybrids)} H, {len(tra_envs)} E)", alpha=0.6),
                  patches.Patch(color=colors['val'], label=f"Validation Set ({len(val_hybrids)} H, {len(val_envs)} E)", alpha=0.6),
                  patches.Patch(color=colors['test'], label=f"Test Set ({len(tes_hybrids)} H, {len(tes_envs)} E)", alpha=0.6)]
ax.legend(handles=legend_patches, loc='upper right', fontsize=12)

ax.set_aspect('auto', adjustable='box')
plt.tight_layout()
plt.show()
# %% Save the data
# genotypes_numeric.to_csv("genotypes_numeric.csv")
with open(f"Formated_Genotypes.pickle", "wb") as f:
    pickle.dump(genotypes_numeric, f)
with open(f"Formated_Environment.pickle", "wb") as f:
    pickle.dump(environments, f)

# # # ## 3. Data Preprocessing Module
# X_tra, Y_tra = data_tra[['Environment', 'Hybrid']].values, data_tra[['Yield', 'Moisture']].values
# X_val, Y_val = data_val[['Environment', 'Hybrid']].values, data_val[['Yield', 'Moisture']].values
# feature_G = genotypes_numeric
# feature_E = environments