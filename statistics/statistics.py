### Statistics

# imports
import pandas as pd; pd.set_option('display.max_columns', None)
import scipy.stats
from tqdm.notebook import tqdm
import pingouin as pg

# ---------- Phonetic Clustering Scores ----------
# 1-sample t-tests against chance, FDR correction (Benjamini-Hochberg) within experiment type
def pcs_statistics(pcs_H_data_bsa, pcs_HS_data_bsa, pcs_HR_data_bsa, pcs_J_data_bsa, pcs_JFL_data_bsa):
    stats = []

    # H
    for et, data in pcs_H_data_bsa.groupby('exp_type'):
        res = scipy.stats.ttest_1samp(data.pcs, popmean=0, nan_policy='omit', alternative='two-sided')
        stats.append(('H', et, res.statistic, res.pvalue, res.df))
    
    # HS
    for et, data in pcs_HS_data_bsa.groupby('exp_type'):
        res = scipy.stats.ttest_1samp(data.pcs, popmean=0, nan_policy='omit', alternative='two-sided')
        stats.append(('HS', et, res.statistic, res.pvalue, res.df))
    
    # HR  
    for et, data in pcs_HR_data_bsa.groupby('exp_type'):
        res = scipy.stats.ttest_1samp(data.pcs, popmean=0, nan_policy='omit', alternative='two-sided')
        stats.append(('HR', et, res.statistic, res.pvalue, res.df))

    # J
    for et, data in pcs_J_data_bsa.groupby('exp_type'):
        res = scipy.stats.ttest_1samp(data.pcs, popmean=0.5, nan_policy='omit', alternative='two-sided')
        stats.append(('J', et, res.statistic, res.pvalue, res.df))
        
    # JFL
    for et, data in pcs_JFL_data_bsa.groupby('exp_type'):
        res = scipy.stats.ttest_1samp(data.pcs, popmean=0.5, nan_policy='omit', alternative='two-sided')
        stats.append(('JFL', et, res.statistic, res.pvalue, res.df))

    # save results as dataframe
    stats = pd.DataFrame(stats, columns=['version', 'exp_type', 't_stat', 'p_val', 'dof'])
    
    # FDR correction (within experiment type)
    pcs_stats = []
    for et, stat in stats.groupby('exp_type'):
        stat['p_val_fdr'] = scipy.stats.false_discovery_control(stat.p_val, method='bh')
        pcs_stats.append(stat)
    
    return pd.concat(pcs_stats, ignore_index=True)


# ---------- Temporal and Semantic Clustering Scores ----------
def cs_statistics(cs_data_bsa, cs):
    stats = []

    for et, data in cs_data_bsa.groupby('exp_type'):
        res = scipy.stats.ttest_1samp(data[cs], popmean=0.5, nan_policy='omit', alternative='two-sided')
        stats.append((et, res.statistic, res.pvalue, res.df))

    # save results as dataframe
    stats = pd.DataFrame(stats, columns=['exp_type', 't_stat', 'p_val', 'dof'])
    
    return stats


# ---------- Phonetic-CRL ----------
# linear regression of IRT as a function of psim for each session.  Average slopes across sessions within subject.  1-sample t-test against slope of 0 with two-sided alternative hypothesis.
def psim_crl_v2_statistics(psim_crl_v2_data_tr):
    # linear regression for each session
    psim_crl_v2_lr = []
    for (sub, et, exp, sess, loc, mont), data in psim_crl_v2_data_tr.groupby(['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage']):
        try:
            slope, _, _, _, _ = scipy.stats.linregress(data.psim, data.irt)
            psim_crl_v2_lr.append((sub, et, exp, sess, loc, mont, slope))
        except ValueError:      # all psim identical
            continue

    psim_crl_v2_lr = pd.DataFrame(psim_crl_v2_lr, columns=['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage', 'slope'])
    
    # average across sessions within subject
    psim_crl_v2_lr_bsa = psim_crl_v2_lr.groupby(['subject', 'exp_type', 'experiment'])['slope'].mean().reset_index()
    
    # 1-sample t-test against 0
    res_intrac = scipy.stats.ttest_1samp(psim_crl_v2_lr_bsa.query("exp_type == 'intracranial'").slope, 0, nan_policy='omit', alternative='two-sided')
    res_scalp = scipy.stats.ttest_1samp(psim_crl_v2_lr_bsa.query("exp_type == 'scalp'").slope, 0, nan_policy='omit', alternative='two-sided')
    
    return pd.DataFrame([('intracranial', res_intrac.statistic, res_intrac.pvalue, res_intrac.df), ('scalp', res_scalp.statistic, res_scalp.pvalue, res_scalp.df)], columns=['exp_type', 't_stat', 'p_val', 'dof'])


# ---------- Phonetic Intrusions ----------
# 1-sample t-test against 0 with two-sided alternative hypothesis for the (PLI/ELI - PLI/ELI control).  
# Paired t-test comparing (PLI - PLI control & ELI - ELI control).  
# FDR correction (Benjamini-Hochberg) within experiment type.  
def psim_intr_l_statistics(psim_intr_l_data_bsa):
    stats = []

    for et, data in psim_intr_l_data_bsa.groupby('exp_type'):
        res_pli = scipy.stats.ttest_1samp(data.pli_delta, 0, nan_policy='omit', alternative='two-sided')
        res_eli = scipy.stats.ttest_1samp(data.eli_delta, 0, nan_policy='omit', alternative='two-sided')
        res_intr = scipy.stats.ttest_rel(data.eli_delta, data.pli_delta, nan_policy='omit', alternative='two-sided')

        stats.append((et, 'pli', res_pli.statistic, res_pli.pvalue, res_pli.df))
        stats.append((et, 'eli', res_eli.statistic, res_eli.pvalue, res_eli.df))
        stats.append((et, 'intr', res_intr.statistic, res_intr.pvalue, res_intr.df))

    # save results as dataframe
    stats = pd.DataFrame(stats, columns=['exp_type', 'comparison', 't_stat', 'p_val', 'dof'])

    # FDR correction (within experiment type)
    psim_intr_stats = []
    for et, stat in stats.groupby('exp_type'):
        stat['p_val_fdr'] = scipy.stats.false_discovery_control(stat.p_val, method='bh')
        psim_intr_stats.append(stat)
        
    return pd.concat(psim_intr_stats, ignore_index=True)


# Paired t-tests (PLI-CR, ELI-CR, PLI-ELI)
# FDR correction (Benjamnini-Hochberg) within experiment type
def psim_intr_r_statistics(psim_intr_r_data_bsa):
    # re-organize dataframe
    df = psim_intr_r_data_bsa.pivot(index=['subject', 'exp_type', 'experiment'], columns='resp_type', values='psim')
    df.columns = [f"{col}_psim" for col in df.columns]
    df.reset_index(inplace=True)
    
    stats = []
    for et, data in df.groupby('exp_type'):
        res_pli = scipy.stats.ttest_rel(data.pli_psim, data.cr_psim, nan_policy='omit', alternative='two-sided')
        res_eli = scipy.stats.ttest_rel(data.eli_psim, data.cr_psim, nan_policy='omit', alternative='two-sided')
        res_intr = scipy.stats.ttest_rel(data.eli_psim, data.pli_psim, nan_policy='omit', alternative='two-sided')

        stats.append((et, 'pli', res_pli.statistic, res_pli.pvalue, res_pli.df))
        stats.append((et, 'eli', res_eli.statistic, res_eli.pvalue, res_eli.df))
        stats.append((et, 'intr', res_intr.statistic, res_intr.pvalue, res_intr.df))

    # save results as dataframe
    stats = pd.DataFrame(stats, columns=['exp_type', 'comparison', 't_stat', 'p_val', 'dof'])

    # FDR correction (within experiment type)
    psim_intr_stats = []
    for et, stat in stats.groupby('exp_type'):
        stat['p_val_fdr'] = scipy.stats.false_discovery_control(stat.p_val, method='bh')
        psim_intr_stats.append(stat)
        
    return pd.concat(psim_intr_stats, ignore_index=True)


# repeated measures anova
def psim_intr_r_rm_anova(psim_intr_r_data_tr, exp_type):
    # only subjects with data in all 3 conditions
    subs_balanced = []
    for (sub, et, exp), data in psim_intr_r_data_tr.groupby(['subject', 'exp_type', 'experiment']):
        if all([x in data.resp_type.unique() for x in ['cr', 'pli', 'eli']]):
            subs_balanced.append(sub)

    df = psim_intr_r_data_tr[psim_intr_r_data_tr.subject.isin(subs_balanced)]

    anova_results = pg.rm_anova(dv='psim', within='resp_type', subject='subject', data=df.query("exp_type == @exp_type"))
    return anova_results


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

    # FDR correction (within experiment type)
    p_recall_corrs_stats = []
    for et, stat in stats.groupby('exp_type'):
        stat['p_val_fdr'] = scipy.stats.false_discovery_control(stat.p_val, method='bh')
        p_recall_corrs_stats.append(stat)

    return pd.concat(p_recall_corrs_stats, ignore_index=True)