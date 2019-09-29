import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

DPI = 300


def label_share(share, fp):
    share_norm = share / share.sum()
    fig, ax = plt.subplots()
    bar = sns.barplot(share_norm.index, share_norm.values)
    for idx, p in enumerate(bar.patches):
        bar.annotate('{:.2f}\n({})'.format(share_norm[idx], share[idx]),
                     (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                     ha='center', va='center', color='white', fontsize='large')
    ax.set_xlabel('Label')
    ax.set_ylabel('Share')
    ax.set_title('Label Share')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def corr_matrix(corr, fp):
    fig, ax = plt.subplots()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, vmin=-1, vmax=1, mask=mask,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                linewidths=0.5, cbar=True, square=True, ax=ax)
    ax.set_title('Correlation Matrix')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def confusion_matrix(cm, fp, norm_axis=1):
    """
    [TN, FP]
    [FN, TP]
    """

    cm_norm = cm / cm.sum(axis=norm_axis, keepdims=True)
    TN, FP, FN, TP = map(str, cm.ravel())
    TN_norm, FP_norm, FN_norm, TP_norm = map(lambda x: str(round(x, 2)), cm_norm.ravel())
    annot = np.array([
        [f'TN: {TN}\n({TN_norm})', f'FP: {FP}\n({FP_norm})'],
        [f'FN: {FN}\n({FN_norm})', f'TP: {TP}\n({TP_norm})']
    ])

    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=1,
                annot=annot, fmt='s', annot_kws={'fontsize': 15},
                linewidths=0.2, cbar=True, square=True, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def metric(metrics, fp):
    fig, ax = plt.subplots()
    for idx, data in enumerate(metrics):
        line = ax.plot(data['values'], label='fold{}'.format(idx), zorder=1)[0]
        ax.scatter(data['best_iteration'], data['values'][data['best_iteration'] - 1],
                   s=60, c=[line.get_color()], edgecolors='k', linewidths=1, zorder=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(metrics[0]['name'])
    ax.set_title('Metric History (marker on each line represents the best iteration)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def feature_importance(features, feature_importances, title, fp):
    fig, ax = plt.subplots()
    idxes = np.argsort(feature_importances)[::-1]
    y = np.arange(len(feature_importances))
    ax.barh(y, feature_importances[idxes][::-1], align='center', height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(features[idxes][::-1])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def scores(scores, fp):
    array = np.array([v for v in scores.values()]).reshape((2, 2))
    annot = np.array(['{}: {}'.format(k, round(v, 3)) for k, v in scores.items()]).reshape((2, 2))
    fig, ax = plt.subplots()
    sns.heatmap(array, cmap='Blues', vmin=0, vmax=1,
                annot=annot, fmt='s', annot_kws={'fontsize': 15},
                linewidths=0.1, cbar=True, square=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Average Classification Scores')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def roc_curve(fpr, tpr, fp):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], 'k:')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def pr_curve(pre, rec, fp):
    fig, ax = plt.subplots()
    ax.plot(pre, rec)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Presision')
    ax.set_title('Precision-Recall Curve')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
