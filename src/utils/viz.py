import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

def plot_confusion(cm, classes, out_png):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close(fig)

def plot_pr(y_true_bin, y_score_bin, classes, out_png):
    aps = []
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score_bin[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score_bin[:, i])
        aps.append(ap); ax.plot(recall, precision, label=f"{c[:16]}.. AP={ap:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Per-class PR curves (macro AP={np.mean(aps):.2f})")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close(fig)

def plot_support_bar(class_counts, classes, out_png, top_n=25):
    idx = np.argsort(class_counts)[::-1][:top_n]
    plt.figure(figsize=(10,5))
    plt.bar(range(len(idx)), np.array(class_counts)[idx])
    plt.xticks(range(len(idx)), [classes[i][:18] + (".." if len(classes[i])>18 else "") for i in idx], rotation=60, ha="right")
    plt.ylabel("Validation samples")
    plt.title(f"Top-{top_n} class support (val)")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_topk_bars(top1, top3, top5, out_png):
    plt.figure(figsize=(5,4))
    ks = ["Top-1","Top-3","Top-5"]; vals = [top1*100, top3*100, top5*100]
    plt.bar(ks, vals, color=["#4c78a8","#f58518","#54a24b"])
    plt.ylim(0,100); plt.ylabel("Accuracy (%)")
    for i,v in enumerate(vals): plt.text(i, v+1, f"{v:.1f}%", ha="center")
    plt.title("Top-k accuracy"); plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_roc_micro(y_true_bin, probs, out_png):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"micro ROC (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_classwise_ap(aps, classes, out_png, top_n=25):
    idx = np.argsort(aps)[::-1][:top_n]
    plt.figure(figsize=(10,5))
    plt.bar(range(len(idx)), np.array(aps)[idx])
    plt.xticks(range(len(idx)), [classes[i][:18] + (".." if len(classes[i])>18 else "") for i in idx], rotation=60, ha="right")
    plt.ylabel("Average Precision")
    plt.title(f"Per-class AP (top {top_n})")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
