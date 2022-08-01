
import matplotlib.pyplot as plt
import numpy as np



def main():
    
    nodes = [2, 4, 8, 16]
    time_in_min = [159.5, 36.9, 16.8, 13.3]

    fix, ax = plt.subplots(figsize=(4, 3))

    ax.plot(nodes, time_in_min, 'g', label='KGLiDS', marker="x")
    ax.set_xticks(nodes)
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Time (Min)')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig('experiment_kg_construction_scalability.pdf')
    plt.show()


if __name__ == '__main__':
    main()