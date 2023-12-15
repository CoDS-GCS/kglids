import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('automl_results.csv')

# filter datasets with equal scores

df['diff'] = df['KGpip+KGLiDS'] - df['KGpip+Graph4Code']

df = df[(df['diff'].abs().round(2) > 0.01)]
df['color'] = df['diff'].apply(lambda x: 'mediumseagreen' if x > 0 else 'indianred')
df = df.sort_values('diff')

print(len(df), 'Datasets')

binary = df[df['Task'] == 'binary']
multiclass = df[df['Task'] == 'multi-class']

plt.style.use('seaborn-white')
plt.rcParams["font.size"] = 18

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
fig.patch.set_facecolor('white')
binary_ids = binary['new_ID'].astype(str).tolist()
binary_diffs = binary['diff'].tolist()
multiclass_ids = multiclass['new_ID'].astype(str).tolist()
multiclass_diffs = multiclass['diff'].tolist()


ax1.bar(np.arange(len(multiclass_diffs)), multiclass_diffs, align='center', color=multiclass.color.tolist())
ax1.set_xticks(np.arange(len(multiclass_diffs)), multiclass_ids, rotation=90)
ax1.set_yticks(np.arange(-0.1, max(multiclass_diffs)+0.05, 0.05))
ax1.set_ylabel('F1-Score Absolute Difference', fontsize=25)
ax1.set_xlabel('Multi-class Datasets', fontsize=30)
ax1.grid(linestyle = '--', linewidth = 1, axis='y')


ax2.bar(np.arange(len(binary_diffs)), binary_diffs, align='center', color=binary.color.tolist())
ax2.set_xticks(np.arange(len(binary_diffs)), binary_ids, rotation=90)
ax2.set_yticks(np.arange(-0.06, 0.12, 0.02))
# ax2.set_yticks(np.arange(-0.1, max(multiclass_diffs)+0.05, 0.05))
# ax2.set_ylabel('F1-Score Absolute Difference', fontsize=25)
ax2.set_xlabel('Binary Datasets', fontsize=30)
ax2.grid(linestyle = '--', linewidth = 1, axis='y')
ax2.set_facecolor('white')


plt.tight_layout()
plt.savefig('automl_results_vs_graph4_code.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
