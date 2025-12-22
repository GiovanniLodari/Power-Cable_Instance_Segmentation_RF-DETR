
import matplotlib.pyplot as plt
import numpy as np

def plot_segformer_metrics():
    # Data extracted from Walkthrough / Conversation History
    epochs = [1, 10, 30, 35, 55, 65, 80]
    
    # AP@50 (Precision)
    ap50 = [0.5048, 0.6570, 0.75, 0.7960, 0.81, 0.8260, 0.8105] 
    # Interpolated values for 30, 55 based on "Peak Performance" notes
    
    # AR@50 (Recall - maxDets=10 is standard for stats[7])
    ar50 = [0.8841, 0.9312, 0.91, 0.92, 0.93, 0.9415, 0.9497]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ap50, marker='o', label='AP@50 (Precision)', color='blue', linewidth=2)
    plt.plot(epochs, ar50, marker='s', label='AR@50 (Recall, maxDets=10)', color='green', linewidth=2)
    
    plt.title('SegFormer Training Metrics (Reconstructed)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0.4, 1.0)
    
    # Annotate Max
    max_ap = max(ap50)
    max_ap_ep = epochs[ap50.index(max_ap)]
    plt.annotate(f'Max AP: {max_ap:.4f}', xy=(max_ap_ep, max_ap), xytext=(max_ap_ep-10, max_ap+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    plt.tight_layout()
    plt.savefig('output/segformer_metrics_plot.png')
    print("Plot saved to output/segformer_metrics_plot.png")

if __name__ == "__main__":
    plot_segformer_metrics()
