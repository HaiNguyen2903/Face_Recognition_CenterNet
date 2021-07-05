import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    ar = np.random.rand(50, 1)
    ar2 = np.random.rand(50, 1)
    fig = plt.figure()

    plt.title("Training and Validation Loss")
    plt.plot(ar,'r', label="val")
    plt.plot(ar2, 'b', label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    save_path = 'saved/losses/loss.png'
    fig.savefig(save_path)