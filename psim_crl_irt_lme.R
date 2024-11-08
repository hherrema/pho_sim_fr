# imports
library(lme4)
library(lmerTest)

# ---------- binary representation ----------
dfH <- read.csv('analyses/dataframes/psim_crl_H_data_tr.csv')

# treat categorical variables as factors
dfH$subject <- factor(dfH$subject)

# fit models
mH1 <- lmer(log_crl ~ psim + ssim + abs_lag + outpos + ncr + len_ph +
              psim:ssim + psim:abs_lag + ssim:abs_lag +
              (1 | subject) + (1 | session) + (1 | list), data=dfH)
prams <- getME(mH1, 'theta')
mH1 <- update(mH1, start=prams, control=lmerControl(optimizer='bobyqa'))

mH2 <- lmer(log_crl ~ psim + ssim + abs_lag + outpos + ncr + len_ph +
              psim:ssim + psim:abs_lag + ssim:abs_lag +
              (1 + psim | subject) + (1 | session) + (1 | list), data=dfH)
prams <- getME(mH2, 'theta')
mH2 <- update(mH2, start=prams, control=lmerControl(optimizer='bobyqa'))

# model selection
aicH <- AIC(mH1, mH2)
bicH <- BIC(mH1, mH2)
mH_select <- data.frame(Model = rownames(aicH), dof = aicH$df, AIC = aicH$AIC, BIC = bicH$BIC)
write.csv(mH_select, 'statistics/dataframes/psim_crl_H_ms.csv', row.names=FALSE)

# save out predicted values
dfH$log_crl_pred <- predict(mH2)
write.csv(dfH, 'statistics/dataframes/psim_crl_H_pred.csv', row.names=FALSE)

# save out fixed effects
summaryH <- summary(mH2)
feH <- summaryH$coefficients
write.csv(as.data.frame(feH), 'statistics/dataframes/psim_crl_H_lme.csv', row.names=TRUE)

# confidence intervals
ciH <- confint(mH2, parm='beta_', method='Wald')
write.csv(as.data.frame(ciH), 'statistics/dataframes/psim_crl_H_ci.csv', row.names=TRUE)


# ---------- continuous representation ----------
dfJ <- read.csv('analyses/dataframes/psim_irt_J_data_tr.csv')

# treat categorical variables as factors
dfJ$subject <- factor(dfJ$subject)

# fit models
mJ1 <- lmer(log_irt ~ psim + ssim + abs_lag + outpos + ncr + len_ph +
              psim:ssim + psim:abs_lag + ssim:abs_lag +
              (1 | subject) + (1 | session) + (1 | list), data=dfJ)

mJ2 <- lmer(log_irt ~ psim + ssim + abs_lag + outpos + ncr + len_ph +
              psim:ssim + psim:abs_lag + ssim:abs_lag +
              (1 + psim | subject) + (1 | session) + (1 | list), data=dfJ)
prams <- getME(mJ2, 'theta')
mJ2 <- update(mJ2, start=prams, control=lmerControl(optimizer='bobyqa'))

# model selection
aicJ <- AIC(mJ1, mJ2)
bicJ <- BIC(mJ1, mJ2)
mJ_select <- data.frame(Model = rownames(aicJ), dof = aicJ$df, AIC = aicJ$AIC, BIC = bicJ$BIC)
write.csv(mJ_select, 'statistics/dataframes/psim_irt_J_ms.csv', row.names=FALSE)

# save out predicted values
dfJ$log_irt_pred <- predict(mJ2)
write.csv(dfJ, 'statistics/dataframes/psim_irt_J_pred.csv', row.names=FALSE)

# save out fixed effects
summaryJ <- summary(mJ2)
feJ <- summaryJ$coefficients
write.csv(as.data.frame(feJ), 'statistics/dataframes/psim_irt_J_lme.csv', row.names=TRUE)

# confidence intervals
ciJ <- confint(mJ2, parm='beta_', method='Wald')
write.csv(as.data.frame(ciJ), 'statistics/dataframes/psim_irt_J_ci.csv', row.names=TRUE)
