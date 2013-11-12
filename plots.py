import matplotlib.pyplot as plt
from pyUtils.plots import plot_var


def all_pairs(iterable):
    return [(i, j) for i in iterable for j in iterable if i != j]


def plot_one_curve(loggers, mods, ks, metric=''):
    res = [loggers[i].get_values("score_{}2{}".format(*mods))
           for i, k in enumerate(ks)]
    plot_var(res, x=ks, label='{} -> {}'.format(*mods),
                linewidth=2, marker='o', var_style='fill')


def plot_k_graphs(loggers, ks, title='', metric=''):
    plt.figure()
    for mod1, mod2 in loggers:
        plot_one_curve(loggers[(mod1, mod2)], (mod1, mod2), ks, metric=metric)
        plot_one_curve(loggers[(mod1, mod2)], (mod2, mod1), ks, metric=metric)
    plt.legend()
    plt.title(title)
    plt.show()


COLORS = {('image', 'motion'): 'royalblue',
          ('motion', 'image'): 'orange',
          ('image', 'sound'): 'green',
          ('sound', 'image'): 'magenta',
          ('motion', 'sound'): 'yellow',
          ('sound', 'motion'): 'violet',
          }


def plot_boxes_one_exp(logger, mods):
    ax = plt.gca()
    ax.yaxis.grid(True, color='lightgrey')
    plt.ylim(0, 1)
    plt.title('Trained: {}'.format(', '.join(mods)))
    mod_pairs = all_pairs(mods)
    vals = [logger.get_values("score_{}2{}_cosine".format(mod1, mod2))
            for mod1, mod2 in mod_pairs]
    boxes = plt.boxplot(vals)['boxes']
    for box, mods in zip(boxes, mod_pairs):
        coords = zip(box.get_xdata(), box.get_ydata())
        p = plt.Polygon(coords, facecolor=COLORS[mods], alpha=.8)
        ax.add_patch(p)
    plt.xticks(range(1, 1 + len(mod_pairs)),
                ['{} -> {}'.format(*mod_pair) for mod_pair in mod_pairs],
                rotation=25)


def plot_boxes(loggers2, logger3):
    plt.figure()
    nb_pairs = len(loggers2)
    for i, mods in enumerate(loggers2):
        plt.subplot(1, 2 * nb_pairs, 1 + i)
        log = loggers2[mods]
        plot_boxes_one_exp(log, mods)
    plt.subplot(1, 2, 2)
    plot_boxes_one_exp(logger3, logger3.get_value('modalities'))
    plt.show()
