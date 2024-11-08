# imports
library(lme4)
library(lmerTest)

# clustering and recall probability
df <- read.csv('analyses/dataframes/cl_pr.csv')

# treat categorical variables as factors
df$subject <- factor(df$subject)

# ---------- binary metric ----------

# drop empty rows
dfH <- df[!is.na(df$pcs_H), ]

# fit models
mH1 <- lmer(p_recall ~ pcs_H + scs + tcs + pcs_H:scs + pcs_H:tcs + scs:tcs +
              (1 | subject) + (1 | session) + (1 | list), data=dfH)

mH2 <- lmer(p_recall ~ pcs_H + scs + tcs + pcs_H:scs + pcs_H:tcs + scs:tcs +
              (1 + pcs_H + scs + tcs | subject) + (1 | session) + (1 | list), data=dfH)
prams <- getME(mH2, 'theta')
mH2 <- update(mH2, start=prams, control=lmerControl(optimizer='bobyqa'))

# model selection
aicH <- AIC(mH1, mH2)
bicH <- BIC(mH1, mH2)
mH_select <- data.frame(Model = rownames(aicH), dof = aicH$df, AIC = aicH$AIC, BIC = bicH$BIC)
write.csv(mH_select, 'statistics/dataframes/cl_pr_H_ms.csv', row.names=FALSE)

# save out fixed effects
summaryH <- summary(mH2)
feH <- summaryH$coefficients
write.csv(as.data.frame(feH), 'statistics/dataframes/cl_pr_H_lme.csv', row.names=TRUE)

# confidence intervals
ciH <- confint(mH2, parm='beta_', method='Wald')
write.csv(as.data.frame(ciH), 'statistics/dataframes/cl_pr_H_ci.csv', row.names=TRUE)

# save out random effects
reH <- ranef(mH2)$subject
write.csv(as.data.frame(reH), 'statistics/dataframes/cl_pr_H_re.csv', row.names=TRUE)


# ---------- continuous metric ----------

# drop empty rows
dfJ <- df[!is.na(df$pcs_J), ]

# fit models
mJ1 <- lmer(p_recall ~ pcs_J + scs + tcs + pcs_J:scs + pcs_J:tcs + scs:tcs +
              (1 | subject) + (1 | session) + (1 | list), data=dfJ)

mJ2 <- lmer(p_recall ~ pcs_J + scs + tcs + pcs_J:scs + pcs_J:tcs + scs:tcs +
              (1 + pcs_J + scs + tcs | subject) + (1 | session) + (1 | list), data=dfJ)
prams <- getME(mJ2, 'theta')
mJ2 <- update(mJ2, start=prams, control=lmerControl(optimizer='bobyqa'))

# model selection
aicJ <- AIC(mJ1, mJ2)
bicJ <- BIC(mJ1, mJ2)
mJ_select <- data.frame(Model = rownames(aicJ), dof = aicJ$df, AIC = aicJ$AIC, BIC = bicJ$BIC)
write.csv(mJ_select, 'statistics/dataframes/cl_pr_J_ms.csv', row.names=FALSE)

# save out fixed effects
summaryJ <- summary(mJ2)
feJ <- summaryJ$coefficients
write.csv(as.data.frame(feJ), 'statistics/dataframes/cl_pr_J_lme.csv', row.names=TRUE)

# confidence intervals
ciJ <- confint(mJ2, parm='beta_', method='Wald')
write.csv(as.data.frame(ciJ), 'statistics/dataframes/cl_pr_J_ci.csv', row.names=TRUE)

# save out random effects
reJ <- ranef(mJ2)$subject
write.csv(as.data.frame(reJ), 'statistics/dataframes/cl_pr_J_re.csv', row.names=TRUE)
