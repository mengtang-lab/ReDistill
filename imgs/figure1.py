import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import numpy as np

if __name__ == "__main__":
    info = {
        # "stl10-mobilenetv2": {
        #     "T": {
        #         "accuracy": 85.34,
        #         "peak memory": 750.00,
        #         "model size": 13.50,
        #     },
        #     "S": {
        #         "accuracy": 69.41,
        #         "peak memory": 50.78,
        #         "model size": 13.50,
        #     },
        #     "RED": {
        #         "accuracy": 73.65,
        #         "peak memory": 50.78,
        #         "model size": 14.25,
        #     },
        # },

        # "mobilenetv3": {
        #     "Teacher": {
        #         "accuracy": 83.74,
        #         "peak memory": 140.63,
        #         "model size": 9.75,
        #     },
        #     "Student": {
        #         "accuracy": 71.27,
        #         "peak memory": 30.69,
        #         "model size": 9.75,
        #     },
        #     "KD": {
        #         "accuracy": 71.56,
        #         "peak memory": 30.69,
        #         "model size": 9.75,
        #     },
        #     "FitNet": {
        #         "accuracy": 71.93,
        #         "peak memory": 30.69,
        #         "model size": 9.75,
        #     },
        #     "AT": {
        #         "accuracy": 75.88,
        #         "peak memory": 30.69,
        #         "model size": 9.75,
        #     },
        #     # "SP": {
        #     #     "accuracy": 72.42,
        #     #     "peak memory": 30.69,
        #     #     "model size": 9.75,
        #     # },
        #     # "VID": {
        #     #     "accuracy": 71.83,
        #     #     "peak memory": 30.69,
        #     #     "model size": 9.75,
        #     # },
        #     "AB": {
        #         "accuracy": 72.94,
        #         "peak memory": 30.69,
        #         "model size": 9.75,
        #     },
        #     "FT": {
        #         "accuracy": 72.11,
        #         "peak memory": 30.69,
        #         "model size": 9.75,
        #     },
        #     "NST": {
        #         "accuracy": 76.11,
        #         "peak memory": 30.69,
        #         "model size": 9.75,
        #     },
        #     "RED (ours)": {
        #         "accuracy": 77.31,
        #         "peak memory": 30.69,
        #         "model size": 10.18,
        #     },
        # },
        #
        # "resnext18": {
        #     "Teacher": {
        #         "accuracy": 85.12,
        #         "peak memory": 500.00,
        #         "model size": 21.47,
        #     },
        #     "Student": {
        #         "accuracy": 79.07,
        #         "peak memory": 75.00,
        #         "model size": 21.47,
        #     },
        #     "KD": {
        #         "accuracy": 81.27,
        #         "peak memory": 75.00,
        #         "model size": 21.47,
        #     },
        #     "FitNet": {
        #         "accuracy": 80.21,
        #         "peak memory": 75.00,
        #         "model size": 21.47,
        #     },
        #     "AT": {
        #         "accuracy": 82.90,
        #         "peak memory": 75.00,
        #         "model size": 21.47,
        #     },
        #     # "SP": {
        #     #     "accuracy": 77.59,
        #     #     "peak memory": 75.00,
        #     #     "model size": 21.47,
        #     # },
        #     # "VID": {
        #     #     "accuracy": 76.99,
        #     #     "peak memory": 75.00,
        #     #     "model size": 21.47,
        #     # },
        #     "AB": {
        #         "accuracy": 81.42,
        #         "peak memory": 75.00,
        #         "model size": 21.47,
        #     },
        #     "FT": {
        #         "accuracy": 80.64,
        #         "peak memory": 75.00,
        #         "model size": 21.47,
        #     },
        #     "NST": {
        #         "accuracy": 81.89,
        #         "peak memory": 75.00,
        #         "model size": 21.47,
        #     },
        #     "RED (ours)": {
        #         "accuracy": 84.80,
        #         "peak memory": 75.00,
        #         "model size": 25.56,
        #     },
        # },

        "imagenet-resnet18": {
            "Teacher": {
                "accuracy": 69.75,
                "peak memory": 3.83,
                "model size": 44.63,
            },
            "Vanilla Student": {
                "accuracy": 61.79,
                "peak memory": 0.77,
                "model size": 44.63,
            },
            "ReviewKD (CVPR'21)": {
                "accuracy": 63.30,
                "peak memory": 0.77,
                "model size": 44.63,
            },
            "MLLD (CVPR'23)": {
                "accuracy": 64.66,
                "peak memory": 0.77,
                "model size": 44.63,
            },
            "RED (ours)": {
                "accuracy": 65.23,
                "peak memory": 0.77,
                "model size": 48.72,
            },
        },

        "imagenet-resnet50": {
            "Teacher": {
                "accuracy": 76.13,
                "peak memory": 9.19,
                "model size": 97.70,
            },
            "Vanilla Student": {
                "accuracy": 69.50,
                "peak memory": 2.30,
                "model size": 97.70,
            },
            "ReviewKD (CVPR'21)": {
                "accuracy": 70.22,
                "peak memory": 2.30,
                "model size": 97.70,
            },
            "MLLD (CVPR'23)": {
                "accuracy": 70.77,
                "peak memory": 2.30,
                "model size": 97.70,
            },
            "RED (ours)": {
                "accuracy": 73.23,
                "peak memory": 2.30,
                "model size": 160.44,
            },
        },
        #
        # "imagenet-mobilenetv2": {
        #     "T": {
        #         "accuracy": 78.32,
        #         "peak memory": 918.75,
        #         "model size": 230.20,
        #     },
        #     "S": {
        #         "accuracy": 62.65,
        #         "peak memory": 114.84,
        #         "model size": 16.23,
        #     },
        #     "RED": {
        #         "accuracy": 68.89,
        #         "peak memory": 229.69,
        #         "model size": 32.73,
        #     },
        # }
    }

    methods = ["Teacher", "Vanilla Student", "ReviewKD (CVPR'21)", "MLLD (CVPR'23)", "RED (ours)"]
    # methods = ["Teacher", "Student", "KD", "FitNet", "AT", "AB", "FT", "NST", "RED (ours)"]
    # methods = ["Teacher", "Student", "KD", "FitNet", "AT", "SP", "VID", "AB", "FT", "NST", "RED (ours)"]

    plt.figure(figsize=(4, 4))
    plt.grid(visible=True)
    markers = ["+", "x"]
    # colors = ["r", "g", "b", "y", "c", "m"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', 'red', '#7f7f7f', '#bcbd22', '#17becf',]
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'm', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'red']
    for j, method in enumerate(methods):
        peakmem_acc = []
        for i, (model_name, model_info) in enumerate(info.items()):
            acc = model_info[method]["accuracy"]
            peak_mem = model_info[method]["peak memory"]
            model_size = model_info[method]["model size"]
            if j == 0:
                plt.scatter(peak_mem, acc, s=30, alpha=1.0, marker=markers[i], c='k', label=model_name.split('-')[1])
            else:
                plt.scatter(peak_mem, acc, s=30, alpha=1.0, marker=markers[i], c='k')

    # l1 = plt.legend(['resnet18', 'resnet50'], loc="upper left")

    plegend = []
    for j, method in enumerate(methods):
        peakmem_acc = []
        for i, (model_name, model_info) in enumerate(info.items()):
            acc = model_info[method]["accuracy"]
            peak_mem = model_info[method]["peak memory"]
            model_size = model_info[method]["model size"]
            peakmem_acc.append([peak_mem, acc])
            a = plt.scatter(peak_mem, acc, s=model_size * 2, alpha=0.5, marker='o', c=colors[j], edgecolors=colors[j], linewidths=2)
            if i == 0: plegend.append(a)
        peakmem_acc = np.array(peakmem_acc)
        # if method in ["Teacher", "Student", "RED (ours)"]:
        # plt.plot(peakmem_acc[:,0], peakmem_acc[:,1], c=colors[j], linestyle='dashed', linewidth=0.8)

    for j, method in enumerate(methods):
        peakmem_acc = []
        for i, (model_name, model_info) in enumerate(info.items()):
            acc = model_info[method]["accuracy"]
            peak_mem = model_info[method]["peak memory"]
            model_size = model_info[method]["model size"]
            plt.scatter(peak_mem, acc, s=30, alpha=1.0, marker=markers[i], c='k')

    plt.legend(plegend, methods, loc="lower right")
    # plt.gca().add_artist(l1)
    plt.xlabel("Peak Memory (MB)")
    plt.ylabel("Accuracy")
    plt.title("ImageNet dataset")

    plt.savefig('./figure1.jpg', dpi=600)
    # plt.savefig('./figure1.eps', dpi=600, format='eps')