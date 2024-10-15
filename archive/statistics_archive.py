# imports
import pandas as pd; pd.set_option('display.max_columns', None)
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm.notebook import tqdm

def psim_intr_v1_statistics(psim_intr_v1_data_bsa):
    stats = []

    for et, data in psim_intr_v1_data_bsa.groupby('exp_type'):
        res_pli = scipy.stats.ttest_rel(data.psim_pli, data.psim_pli_control, nan_policy='omit', alternative='two-sided')
        res_eli = scipy.stats.ttest_rel(data.psim_eli, data.psim_eli_control, nan_policy='omit', alternative='two-sided')
        res_intr = scipy.stats.ttest_rel(data.psim_pli, data.psim_eli, nan_policy='omit', alternative='two-sided')

        stats.append((et, 'pli', res_pli.statistic, res_pli.pvalue, res_pli.df))
        stats.append((et, 'eli', res_eli.statistic, res_eli.pvalue, res_eli.df))
        stats.append((et, 'intr', res_intr.statistic, res_intr.pvalue, res_intr.df))

    # save results as dataframe
    stats = pd.DataFrame(stats, columns=['exp_type', 'comparison', 't_stat', 'p_val', 'dof'])

    # FDR correction
    stats['p_val_fdr'] = scipy.stats.false_discovery_control(stats.p_val, method='bh')
    
    return stats

def psim_intr_r_rm_anova(psim_intr_r_data_tr, exp_type):
    # only subjects with data in all 3 conditions
    subs_balanced = []
    for (sub, et, exp), data in psim_intr_r_data_tr.groupby(['subject', 'exp_type', 'experiment']):
        if all([x in data.resp_type.unique() for x in ['cr', 'pli', 'eli']]):
            subs_balanced.append(sub)

    df = psim_intr_r_data_tr[psim_intr_r_data_tr.subject.isin(subs_balanced)]

    anova_results = pg.rm_anova(dv='psim', within='resp_type', subject='subject', data=df.query("exp_type == @exp_type"))
    return anova_results