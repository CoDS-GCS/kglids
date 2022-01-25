import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


labels = ['Parameter Names']
kglids = [2942]
graph4code = [300]

x = np.array([0.6])  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(4,4))
rects1 = ax.bar(x - 0.0125 - width/2, kglids, width, label='KGLiDS')
rects2 = ax.bar(x + 0.0125 + width/2, graph4code, width, label='GraphGen4Code')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Statement with Correct Parameters')
# ax.set_xticks(x, labels)
ax.legend(loc='upper right')

ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=1)

fig.tight_layout()

plt.savefig('experiment_pipeline_accuracy.pdf')
plt.show()
