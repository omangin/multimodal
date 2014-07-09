from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .lib.plot import boxplot, plot_var, legend


COLORS = {'motion': '#006EB8',
          'image': 'orange',
          'sound': '#3C8031',
          }

PAIRS_COLORS = {('image', 'motion'): '#006EB8',
                ('motion', 'image'): 'orange',
                ('image', 'sound'): '#3C8031',
                ('sound', 'image'): 'violet',
                ('motion', 'sound'): 'Red',
                ('sound', 'motion'): '#FBB982',
                }


def figure(**kwargs):
    if 'frameon' not in kwargs:
        kwargs['frameon'] = False
    return plt.figure(**kwargs)


def get_color_for_modalities(mods1, mods2, colors=None):
    if colors == 'pairs':
        return PAIRS_COLORS[mods1[0], mods2[0]]
    else:
        return COLORS[mods2[0]]


def mod_to_mod_str(mod1, mod2):
    return '{} $\\rightarrow$ {}'.format(mod1, mod2)


def all_pairs(iterable):
    return [([i], [j]) for i in iterable for j in iterable if i != j]


def combinations(iterable):
    pairs = all_pairs(iterable)
    two_to_one = []
    for i in iterable:
        two_to_one.append(([j for j in iterable if j != i], [i]))
    return pairs + two_to_one


def plot_one_curve(loggers, mods, ks, metric='', linestyle='-',
                   var_style='fill'):
    res = [loggers[i].get_values("score_{}2{}".format('_'.join(mods[0]),
                                                      '_'.join(mods[1])))
           for i, k in enumerate(ks)]
    return plot_var(res, x=ks, label=mod_to_mod_str(*mods),
                    color=get_color_for_modalities(*mods, colors='pairs'),
                    linewidth=2, marker='o', var_style=var_style,
                    linestyle=linestyle)


def plot_k_graphs(loggers, ks, title='', metric=''):
    figure()
    for mod1, mod2 in loggers:
        plot_one_curve(loggers[(mod1, mod2)], ([mod1], [mod2]), ks,
                       metric=metric)
        plot_one_curve(loggers[(mod1, mod2)], ([mod2], [mod1]), ks,
                       metric=metric)
    legend()
    plt.title(title)
    plt.show()


def plot_2k_graphs(loggers, ks, title='', metric=''):
    fig = figure()
    lines = []
    for (logs, linestyle) in zip(loggers, ['-', '--']):
        lines.append({})
        for mod1, mod2 in logs:
            l1 = plot_one_curve(logs[(mod1, mod2)], ([mod1], [mod2]), ks,
                                metric=metric, linestyle=linestyle,
                                var_style='bar')
            l2 = plot_one_curve(logs[(mod1, mod2)], ([mod2], [mod1]), ks,
                                metric=metric, linestyle=linestyle,
                                var_style='bar')
            lines[-1][mod_to_mod_str(mod1, mod2)] = l1
            lines[-1][mod_to_mod_str(mod2, mod1)] = l2
    legend1 = legend(lines[0].values(), lines[0].keys(), loc=4)
    legend([Line2D(range(5), range(5), color='black', linestyle='-'),
            Line2D(range(5), range(5), color='black', linestyle='--')],
           ['Trained on two modalities.', 'Trained on all three modalities.'])
    ax = plt.gca()
    ax.add_artist(legend1)
    ax.set_xlabel('k')
    ax.set_ylabel('Cross-modal association score')
    plt.title(title)
    return fig


def plot_boxes_one_exp(logger, mods_to_mods, colors=None, xticks=False):
    ax = plt.gca()
    ax.yaxis.grid(True, color='lightgrey')
    plt.ylim(0, 1)
    vals = [logger.get_values("score_{}2{}_cosine".format('_'.join(mods1),
                                                          '_'.join(mods2)))
            for mods1, mods2 in mods_to_mods]
    boxes = boxplot(vals, xticklabels=[])['boxes']
    polygons = []
    for box, mods in zip(boxes, mods_to_mods):
        coords = zip(box.get_xdata(), box.get_ydata())
        fcolor = get_color_for_modalities(*mods, colors=colors)
        p = plt.Polygon(coords, facecolor=fcolor, alpha=.8)
        ax.add_patch(p)
        polygons.append(p)
    labels = ['{} $\\rightarrow$ {}'.format(', '.join(mod1), ', '.join(mod2))
              for mod1, mod2 in mods_to_mods]
    if xticks:
        plt.xticks(range(1, 1 + len(mods_to_mods)),
                   labels, rotation=25, ha='right')
    return polygons, labels


def plot_boxes_one_exp_pairs(logger, mods, colors=None, xticks=False):
    return plot_boxes_one_exp(logger, all_pairs(mods), colors=colors,
                              xticks=xticks)


def plot_boxes(loggers2, logger3):
    fig = figure()
    nb_pairs = len(loggers2)
    for i, mods in enumerate(loggers2):
        if i == 0:
            ax1 = plt.subplot(1, 2 * nb_pairs, 1 + i)
            ax1.set_ylabel('Cross-modal association score')
        else:
            ax = plt.subplot(1, 2 * nb_pairs, 1 + i, sharey=ax1)
            ax.label_outer()
            ax.spines['left'].set_visible(False)
        log = loggers2[mods]
        plot_boxes_one_exp_pairs(log, mods, colors='pairs')
        plt.title(', '.join(mods))
    ax = plt.subplot(1, 2, 2, sharey=ax1)
    ax.label_outer()
    ax.spines['left'].set_visible(False)
    mods = logger3.get_value('modalities')
    polygons, labels = plot_boxes_one_exp_pairs(logger3, mods, colors='pairs')
    legend(polygons, labels, fig=fig,  ncol=3, loc='lower center',
           bbox_to_anchor=(.45, .05))
    plt.title(', '.join(mods))
    return fig


def plot_boxes_all_mods(logger3):
    fig = figure()
    mods = logger3.get_value('modalities')
    polygons, labels = plot_boxes_one_exp(logger3, combinations(mods))
    plt.gca().set_ylabel('Cross-modal association score')
    plt.title(', '.join(mods))
    legend(polygons, labels, ncol=3, loc='lower center')
    return fig


def plot_boxes_by_feats(loggers):
    fig = figure()
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
        polygons, labels = plot_boxes_one_exp_pairs(log, mods, colors='pairs')
        plt.title('\n'.join(wrap(', '.join(feats).replace('_', '-'), 15)),
                  fontsize='x-small')
    ax1.set_ylabel('Cross-modal association score')
    legend(polygons, labels, fig=fig,  ncol=3, loc='lower center',
           bbox_to_anchor=(.4, .045))
    return fig
