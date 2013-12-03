from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .lib.plot import boxplot, plot_var, legend


COLORS = {('image', 'motion'): '#006EB8',
          ('motion', 'image'): 'orange',
          ('image', 'sound'): '#3C8031',
          ('sound', 'image'): 'violet',
          ('motion', 'sound'): 'Red',
          ('sound', 'motion'): '#FBB982',
          }


def mod_to_mod_str(mod1, mod2):
    return '{} $\\rightarrow$ {}'.format(mod1, mod2)


def all_pairs(iterable):
    return [(i, j) for i in iterable for j in iterable if i != j]


def plot_one_curve(loggers, mods, ks, metric='', linestyle='-',
                   var_style='fill'):
    res = [loggers[i].get_values("score_{}2{}".format(*mods))
           for i, k in enumerate(ks)]
    return plot_var(res, x=ks, label=mod_to_mod_str(*mods),
                    color=COLORS[mods], linewidth=2, marker='o',
                    var_style=var_style, linestyle=linestyle)


def plot_k_graphs(loggers, ks, title='', metric=''):
    plt.figure()
    for mod1, mod2 in loggers:
        plot_one_curve(loggers[(mod1, mod2)], (mod1, mod2), ks, metric=metric)
        plot_one_curve(loggers[(mod1, mod2)], (mod2, mod1), ks, metric=metric)
    legend()
    plt.title(title)
    plt.show()


def plot_2k_graphs(loggers, ks, title='', metric=''):
    plt.figure()
    lines = []
    for (logs, linestyle) in zip(loggers, ['-', '--']):
        lines.append({})
        for mod1, mod2 in logs:
            l1 = plot_one_curve(logs[(mod1, mod2)], (mod1, mod2), ks,
                                metric=metric, linestyle=linestyle,
                                var_style='bar')
            l2 = plot_one_curve(logs[(mod1, mod2)], (mod2, mod1), ks,
                                metric=metric, linestyle=linestyle,
                                var_style='bar')
            lines[-1][mod_to_mod_str(mod1, mod2)] = l1
            lines[-1][mod_to_mod_str(mod2, mod1)] = l2
    legend1 = legend(lines[0].values(), lines[0].keys(), loc=4)
    legend([Line2D(range(3), range(3), color='black', linewidth=2,
                   linestyle='-'),
            Line2D(range(3), range(3), color='black', linewidth=2,
                   linestyle='--')],
           ['Trained on two modalities.', 'Trained on all three modalities.'])
    plt.gca().add_artist(legend1)
    plt.title(title)
    plt.show()


def plot_boxes_one_exp(logger, mods):
    ax = plt.gca()
    ax.yaxis.grid(True, color='lightgrey')
    plt.ylim(0, 1)
    mod_pairs = all_pairs(mods)
    vals = [logger.get_values("score_{}2{}_cosine".format(mod1, mod2))
            for mod1, mod2 in mod_pairs]
    boxes = boxplot(vals)['boxes']
    for box, mods in zip(boxes, mod_pairs):
        coords = zip(box.get_xdata(), box.get_ydata())
        p = plt.Polygon(coords, facecolor=COLORS[mods], alpha=.8)
        ax.add_patch(p)
    labels = ['{} $\\rightarrow$ {}'.format(*mod_pair)
              for mod_pair in mod_pairs]
    plt.xticks(range(1, 1 + len(mod_pairs)), labels, rotation=25, ha='right')


def plot_boxes(loggers2, logger3):
    plt.figure()
    nb_pairs = len(loggers2)
    for i, mods in enumerate(loggers2):
        if i == 0:
            ax1 = plt.subplot(1, 2 * nb_pairs, 1 + i)
        else:
            ax = plt.subplot(1, 2 * nb_pairs, 1 + i, sharey=ax1)
            ax.label_outer()
            ax.spines['left'].set_visible(False)
        log = loggers2[mods]
        plot_boxes_one_exp(log, mods)
        plt.title(', '.join(mods))
    ax = plt.subplot(1, 2, 2, sharey=ax1)
    ax.label_outer()
    ax.spines['left'].set_visible(False)
    plot_boxes_one_exp(logger3, logger3.get_value('modalities'))
    plt.title(', '.join(mods))
    plt.show()


def plot_boxes_by_feats(loggers):
    plt.figure()
    mods = ('image', 'sound')
    nb_feats = len(loggers)
    for i, feats in enumerate(loggers):
        if i == 0:
            ax1 = plt.subplot(1, nb_feats, 1 + i)
        else:
            ax = plt.subplot(1, nb_feats, 1 + i, sharey=ax1)
            ax.label_outer()
            ax.spines['left'].set_visible(False)
        log = loggers[feats]
        plot_boxes_one_exp(log, mods)
        plt.title('\n'.join(wrap(', '.join(feats).replace('_', '-'), 15)))
    plt.show()
