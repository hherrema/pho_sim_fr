### Statistics

# imports
import pandas as pd; pd.set_option('display.max_columns', None)
import scipy.stats
from tqdm.notebook import tqdm
import pingouin as pg

# ---------- Phonetic Clustering Scores ----------
# 1-sample t-tests against chance, FDR correction (Benjamini-Hochberg)
def pcs_statistics(pcs_H_data_bsa, pcs_HS_data_bsa, pcs_HR_data_bsa, pcs_J_data_bsa, pcs_JFL_data_bsa):
    stats = []

    # H
    res = scipy.stats.ttest_1samp(pcs_H_data_bsa.pcs, popmean=0, nan_policy='omit', alternative='two-sided')
    stats.append(('H', res.df, res.statistic, res.pvalue))

    # HS
    res = scipy.stats.ttest_1samp(pcs_HS_data_bsa.pcs, popmean=0, nan_policy='omit', alternative='two-sided')
    stats.append(('HS', res.df, res.statistic, res.pvalue))

    # HR
    res = scipy.stats.ttest_1samp(pcs_HR_data_bsa.pcs, popmean=0, nan_policy='omit', alternative='two-sided')
    stats.append(('HR', res.df, res.statistic, res.pvalue))

    res = scipy.stats.ttest_1samp(pcs_J_data_bsa.pcs, popmean=0.5, nan_policy='omit', alternative='two-sided')
    stats.append(('J', res.df, res.statistic, res.pvalue))

    res = scipy.stats.ttest_1samp(pcs_JFL_data_bsa.pcs, popmean=0.5, nan_policy='omit', alternative='two-sided')
    stats.append(('JFL', res.df, res.statistic, res.pvalue))
    
    # save as dataframe
    stats = pd.DataFrame(stats, columns=['metric', 'dof', 't_stat', 'p_val'])

    # FDR correction
    stats['p_val_fdr'] = scipy.stats.false_discovery_control(stats.p_val, method='bh')
    
    return stats


# ---------- Temporal and Semantic Clustering Scores ----------
# 1-sample t-test against chance with two-sided alternative hypothesis
def cs_statistics(cs_data_bsa, cs):
    res = scipy.stats.ttest_1samp(cs_data_bsa[cs], popmean=0.5, nan_policy='omit', alternative='two-sided')
    
    return pd.DataFrame([(cs, res.df, res.statistic, res.pvalue)], columns=['score', 'dof', 't_stat', 'p_val'])


# ---------- Phonetic Intrusions ----------
# 1-sample t-test against 0 with two-sided alternative hypothesis (PLI/ELI - PLI/ELI control)
# Paired t-test with two-sided alternative hypothsis (ELI - PLI)
# FDR correction (Benjamini-Hochberg)
def psim_intr_l_statistics(psim_intr_l_H_data_bsa, psim_intr_l_J_data_bsa):
    stats = []
    
    # H
    res_pli = scipy.stats.ttest_1samp(psim_intr_l_H_data_bsa.pli_delta, 0, nan_policy='omit', alternative='two-sided')
    res_eli = scipy.stats.ttest_1samp(psim_intr_l_H_data_bsa.eli_delta, 0, nan_policy='omit', alternative='two-sided')
    res_intr = scipy.stats.ttest_rel(psim_intr_l_H_data_bsa.eli_psim, psim_intr_l_H_data_bsa.pli_psim, nan_policy='omit', alternative='two-sided')
    
    stats.append(('H', 'pli', res_pli.df, res_pli.statistic, res_pli.pvalue))
    stats.append(('H', 'eli', res_eli.df, res_eli.statistic, res_eli.pvalue))
    stats.append(('H', 'intr', res_intr.df, res_intr.statistic, res_intr.pvalue))
    
    # J
    res_pli = scipy.stats.ttest_1samp(psim_intr_l_J_data_bsa.pli_delta, 0, nan_policy='omit', alternative='two-sided')
    res_eli = scipy.stats.ttest_1samp(psim_intr_l_J_data_bsa.eli_delta, 0, nan_policy='omit', alternative='two-sided')
    res_intr = scipy.stats.ttest_rel(psim_intr_l_J_data_bsa.eli_psim, psim_intr_l_J_data_bsa.pli_psim, nan_policy='omit', alternative='two-sided')
    
    stats.append(('J', 'pli', res_pli.df, res_pli.statistic, res_pli.pvalue))
    stats.append(('J', 'eli', res_eli.df, res_eli.statistic, res_eli.pvalue))
    stats.append(('J', 'intr', res_intr.df, res_intr.statistic, res_intr.pvalue))
    
    # save as dataframe
    stats = pd.DataFrame(stats, columns=['metric', 'comparison', 'dof', 't_stat', 'p_val'])
    
    # FDR correction
    stats['p_val_fdr'] = scipy.stats.false_discovery_control(stats.p_val, method='bh')
    
    return stats


# repeated measures ANOVA (CR, PLI, ELI)
# subsequent pairwise tests (CR-PLI, CR-ELI, ELI-PLI) with FDR correction (Benjamini-Hochberg)
def psim_intr_r_rm_anova(psim_intr_r_data_bsa):
    # only subjects with data in all 3 conditions
    subs_balanced = []
    for sub, data in psim_intr_r_data_bsa.groupby(['subject']):
        if all([x in data.resp_type.unique() for x in ['cr', 'pli', 'eli']]):
            subs_balanced.append(sub[0])

    df = psim_intr_r_data_bsa[psim_intr_r_data_bsa.subject.isin(subs_balanced)].query("resp_type != 'control'")
    anova_results = pg.rm_anova(data=df, dv='psim', within='resp_type', subject='subject')
    return anova_results, df