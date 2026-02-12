import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    def __init__(self, interval, range, explain, depend):
        self.interval = interval
        self.range = range
        self.explain = explain
        self.depend = depend
        self.ax.set_xlim(0, range)

    def plot_sigmoid(self, w, b):
        x = np.linspace(0, self.range, 200)
        y = 1 / (1 + np.exp(-w * x - b))
        self.ax.plot(x, y)
        
    def show(self, w, b):
        self.ax.cla()
        self.ax.set_xlabel("explanation valuable")
        self.ax.set_ylabel("dependent valuable")
        self.ax.set_xlim(0, self.range)
        self.ax.scatter(self.explain, self.depend, c='blue', alpha=0.6)
        self.plot_sigmoid(w, b)
        plt.grid(True)
        plt.pause(self.interval)

if __name__ == "__main__":
    """
    explain = np.array([1, 8, 1, 6, 7, 2])
    depend = np.array([0, 1, 0, 1, 1, 0])
    plotter = Plotter(1, 10, explain, depend)
    """
    from generator import Generator
    g = Generator()
    explain, depend = g.generate_single_val_data(100, 100, 30, 5)
    plotter = Plotter(1, 100, explain, depend)
    
    for _ in range(5):
        plotter.show(5, 4)