import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from scipy import interpolate, integrate



#@title Compute Abroca
def compute_roc(y_scores, y_true):
    """
    Function to compute the Receiver Operating Characteristic (ROC) curve for a set of predicted probabilities and the true class labels.
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns FPR and TPR values
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    return fpr, tpr


def compute_auc(y_scores, y_true):
    """
    Function to Area Under the Receiver Operating Characteristic Curve (AUC)
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns AUC value
    """
    auc = metrics.roc_auc_score(y_true, y_scores)
    return auc


def interpolate_roc_fun(fpr, tpr, n_grid):
    """
    Function to Use interpolation to make approximate the Receiver Operating Characteristic (ROC) curve along n_grid equally-spaced values.
    fpr - vector of false positive rates computed from compute_roc
    tpr - vector of true positive rates computed from compute_roc
    n_grid - number of approximation points to use (default value of 10000 more than adequate for most applications) (numeric)
    Returns  a list with components x and y, containing n coordinates which interpolate the given data points according to the method (and rule) desired
    """
    roc_approx = interpolate.interp1d(x=fpr, y=tpr)
    x_new = np.linspace(0, 1, num=n_grid)
    y_new = roc_approx(x_new)
    return x_new, y_new


def slice_plot(
    np_roc_fpr,
    p_roc_fpr,
    np_roc_tpr,
    p_roc_tpr,
    np_group_name,
    p_group_name,
    fout="./slice_plot.png",
    value=0.0
):
    """
    Function to create a 'slice plot' of two roc curves with area between them (the ABROCA region) shaded.
    np_roc_fpr, p_roc_fpr - FPR of np and p groups
    np_roc_tpr, p_roc_tpr - TPR of np and p groups
    np_group_name - (optional) - np group display name on the slice plot
    p_group_name - (optional) - p group display name on the slice plot
    fout - (optional) -  File name (including directory) to save the slice plot generated
    No return value; displays slice plot & file is saved 
    """
    plt.figure(1, figsize=(5, 4))
    title = "ABROCA = " + str(value)
    plt.title(title)
    plt.xlabel("False Positive Rate",fontweight='bold')
    plt.ylabel("True Positive Rate",fontweight='bold')
    plt.ylim((-0.04, 1.04))
    plt.plot(
        np_roc_fpr,
        np_roc_tpr,
        #label="{o} - Baseline".format(o=np_group_name),
        label="{o}".format(o=np_group_name),
        linestyle="-",
        color="r",
    )
    plt.plot(
        p_roc_fpr,
        p_roc_tpr,
        #label="{o} - Comparison".format(o=p_group_name),
        label="{o}".format(o=p_group_name),
        linestyle="-",
        color="b",
    )
    plt.fill(
        np_roc_fpr.tolist() + np.flipud(p_roc_fpr).tolist(),
        np_roc_tpr.tolist() + np.flipud(p_roc_tpr).tolist(),
        "gray",
    )
    plt.legend()
    plt.savefig(fout,bbox_inches = "tight")
    plt.show()


def compute_abroca(
    df,
    pred_col,
    label_col,
    p_attr_col,
    np_p_attr_val,
    n_grid=10000,
    plot_slices=False,
    lb=0,
    ub=1,
    limit=1000,
    file_name="slice_image.png",
    np_group_name = 'Male',
    p_group_name = 'Female'
):
    # Compute the value of the abroca statistic.
    """
    df - dataframe containing colnames matching pred_col, label_col and p_attr_col
    pred_col - name of column containing predicted probabilities (string)
    label_col - name of column containing true labels (should be 0,1 only) (string)
    p_attr_col - name of column containing p attribute (should be binary) (string)
    np_p_attr_val name of 'np' group with respect to p attribute (string)
    n_grid (optional) - number of grid points to use in approximation (numeric) (default of 10000 is more than adequate for most cases)
    plot_slices (optional) - if TRUE, ROC slice plots are generated and saved to file_name (boolean)
    lb (optional) - Lower limit of integration (use -numpy.inf for -infinity) Default is 0
    ub (optional) - Upper limit of integration (use -numpy.inf for -infinity) Default is 1
    limit (optional) - An upper bound on the number of subintervals used in the adaptive algorithm.Default is 1000
    file_name (optional) - File name (including directory) to save the slice plot generated
    Returns Abroca value
    """
    if df[pred_col].between(0, 1, inclusive='both').any():
        pass
    else:
        print("predictions must be in range [0,1]")
        exit(1)
    if len(df[label_col].value_counts()) == 2:
        pass
    else:
        print("The label column should be binary")
        exit(1)
    if len(df[p_attr_col].value_counts()) == 2:
        pass
    else:
        print("The p attribute column should be binary")
        exit(1)
    # initialize data structures
    # slice_score = 0
    prot_attr_values = df[p_attr_col].value_counts().index.values
    fpr_tpr_dict = {}

    # compute roc within each group of pa_values
    for pa_value in prot_attr_values:
        if pa_value != np_p_attr_val:
            p_p_attr_val = pa_value
        pa_df = df[df[p_attr_col] == pa_value]
        fpr_tpr_dict[pa_value] = compute_roc(pa_df[pred_col], pa_df[label_col])

    # compare p to np class; accumulate absolute difference btw ROC curves to slicing statistic
    np_roc_x, np_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[np_p_attr_val][0],
        fpr_tpr_dict[np_p_attr_val][1],
        n_grid,
    )
    p_roc_x, p_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[p_p_attr_val][0],
        fpr_tpr_dict[p_p_attr_val][1],
        n_grid,
    )

    # use function approximation to compute slice statistic via piecewise linear function
    if list(np_roc_x) == list(p_roc_x):
        f1 = interpolate.interp1d(x=np_roc_x, y=(np_roc_y - p_roc_y))
        f2 = lambda x, acc: abs(f1(x))
        slice, _ = integrate.quad(f2, lb, ub, limit)
    else:
        print("np and p FPR are different")
        exit(1)

    if plot_slices == True:
        slice_plot(
            np_roc_x,
            p_roc_x,
            np_roc_y,
            p_roc_y,
            np_group_name=np_group_name,
            p_group_name=p_group_name,
            fout=file_name,
            value=round(slice,4),
        )

    return 