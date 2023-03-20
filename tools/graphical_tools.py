import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Graphical_tools:
    @staticmethod
    def show_df(df, cols=[], rows=None, title='', xlabel='', ylabel=''):
        if not rows:
            rows = cols

        f = plt.figure(figsize=(19, 15))
        plt.matshow(df)
        plt.xticks(range(len(rows)), rows, fontsize=8)
        plt.yticks(range(len(cols)), cols, fontsize=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title(title, fontsize=16)
        plt.show()


    @staticmethod
    def plot_heatmap(arr: np.array, title="", x_labels=[], y_labels=[]):
        ax = plt.axes()
        sns.heatmap(arr, linewidth=0.5, ax=ax, xticklabels=x_labels, yticklabels=y_labels)
        ax.set_title(title)
        plt.show()

    @staticmethod
    def one_arr(y, x=None, title="", x_label="", y_label=""):
        """
        plot one arr

        :param y: y-axis values
        :param x: x-axis values. if None then generates 0-len(y)
        :param title: the graph title
        :param x_label: x-axis label
        :param y_label: y-axis label
        """
        if x == np.array([]):
            x = np.arange(0, len(y))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    @staticmethod
    def many_arrays(arrays: list, descriptions: list=[], title="", x_label="", y_label=""):
        """
        plot many arrays on one graph

        :param arrays: array of tuples - (x, y)
        :param descriptions: array of the graphs labels
        :param title: the graph title
        :param x_label: x-axis label
        :param y_label: y-axis label
        """
        for i, (x, y) in enumerate(arrays):
            if x == np.array([]):
                x = np.arange(0, len(y))
            if not descriptions:
                plt.plot(x, y)
            else:
                plt.plot(x, y, label=descriptions[i])

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def one_np_file(path, title="", x_label="", y_label=""):
        y = np.load(path)
        Graphical_tools.one_arr(y, title, x_label, y_label)

    @staticmethod
    def many_np_file(paths : list, descriptions : list=[], title="", x_label="", y_label=""):
        arrays = []
        for i, (path_x, path_y) in enumerate(paths):
            y = np.load(path_y)
            if not path_x:
                x = np.arange(0, len(y))
            else:
                x = np.load(path_x)
            arrays.append((x, y))

        Graphical_tools.many_arrays(arrays, descriptions, title, x_label, y_label)
