# imports
import pandas as pd; pd.set_option('display.max_columns', None)
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm.notebook import tqdm


# ---------- Correlations with Recall Probability ----------
# 1-sample t-test against no correlation (0).  FDR correction (Benjamini-Hochberg) within experiment type.
def p_recall_corrs_statistics(p_recall_corrs_bsa):
    stats = []
    for et, data in p_recall_corrs_bsa.groupby('exp_type'):
        if len(data) > 1:
            res_pcs_H = scipy.stats.ttest_1samp(data.r_pcs_H, popmean=0, nan_policy='omit', alternative='two-sided')
            res_pcs_J = scipy.stats.ttest_1samp(data.r_pcs_J, popmean=0, nan_policy='omit', alternative='two-sided')
            res_tcs = scipy.stats.ttest_1samp(data.r_tcs, popmean=0, nan_policy='omit', alternative='two-sided')
            res_scs = scipy.stats.ttest_1samp(data.r_scs, popmean=0, nan_policy='omit', alternative='two-sided')

            stats.append(('pcs_H', et, res_pcs_H.statistic, res_pcs_H.pvalue, res_pcs_H.df))
            stats.append(('pcs_J', et, res_pcs_J.statistic, res_pcs_J.pvalue, res_pcs_J.df))
            stats.append(('tcs', et, res_tcs.statistic, res_tcs.pvalue, res_tcs.df))
            stats.append(('scs', et, res_scs.statistic, res_scs.pvalue, res_scs.df))

    stats = pd.DataFrame(stats, columns=['label', 'exp_type', 't_stat', 'p_val', 'dof'])
    
    """
    # FDR correction (within experiment type)
    p_recall_corrs_stats = []
    for et, stat in stats.groupby('exp_type'):
        stat['p_val_fdr'] = scipy.stats.false_discovery_control(stat.p_val, method='bh')
        p_recall_corrs_stats.append(stat)

    return pd.concat(p_recall_corrs_stats, ignore_index=True)
    """
    
    # FDR correct (across experiment type)
    stats['p_val_fdr'] = scipy.stats.false_discovery_control(stats.p_val, method='bh')
    
    return stats