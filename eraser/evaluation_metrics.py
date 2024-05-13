import os
import matplotlib.pyplot as plt
from arguments import Arguments

class Metrics:

    def __init__(self, loss, acc):
        self.loss = loss
        self.acc = acc
    
    def loss_acc_graph(self, save_dir=None, loss_filename=None,acc_filename=None):
        if save_dir is None:
            save_dir = "evaluation_metrics"  
        if loss_filename is None:
            loss_filename = "loss_graph.png"
        if acc_filename is None:
            acc_filename = "accuracy_graph.png" 

        plt.plot(self.loss['training'], label='Training Loss')
        plt.plot(self.loss['validation'], label='Validation Loss')
        plt.plot(self.loss['testing'], label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss') 
        plt.legend()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Path created successfully')
        save_path = os.path.join(save_dir, loss_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss graph saved at: {save_path}")

        plt.plot(self.acc['training'], label='Training acc')
        plt.plot(self.acc['validation'], label='Validation acc')
        plt.plot(self.acc['testing'], label='Testing acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, acc_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Accuracy graph saved at: {save_path}")
