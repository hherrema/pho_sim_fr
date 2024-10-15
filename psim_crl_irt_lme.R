# imports
library(lme4)
library(lmerTest)

# binary representation
dfH <- read.csv('analyses/dataframes/psim_crl_H_data_tr.csv')

# treat categorical variables as factors
dfH$subject <- factor(dfH$subject)
dfH$exp_type <- factor(dfH$exp_type)

mH1 <- lmer(log_crl ~ psim + ssim + abs_lag + outpos + ncr + len_ph + exp_type + 
              psim:ssim + psim:abs_lag + ssim:abs_lag + psim:exp_type + 
              (1 | subject) + (1 | session) + (1 | list), data=dfH)

mH2 <- lmer(log_crl ~ psim + ssim + abs_lag + outpos + ncr + len_ph + exp_type + 
              psim:ssim + psim:abs_lag + ssim:abs_lag + psim:exp_type +
              (1 | subject) + (1 | session:exp_type) + (1 | list), data=dfH)

mH3 <- lmer(log_crl ~ psim + ssim + abs_lag + outpos + ncr + len_ph + exp_type +
              psim:ssim + psim:abs_lag + ssim:abs_lag + psim:exp_type +
              (1 | subject) + (1 | session:exp_type) + (1 | list:exp_type), data=dfH)

# model selection
aicH <- AIC(mH1, mH2, mH3)
bicH <- BIC(mH1, mH2, mH3)

print(aicH)
print(bicH)

summary(mH2)

# save out AIC, BIC
mH_select <- data.frame(Model = rownames(aicH), dof = aicH$df, AIC = aicH$AIC, BIC = bicH$BIC)
write.csv(mH_select, 'statistics/dataframes/psim_crl_H_ms.csv', row.names=FALSE)

# save out fixed effects
summaryH <- summary(mH2)
feH <- summaryH$coefficients

write.csv(as.data.frame(feH), 'statistics/dataframes/psim_crl_H_lme.csv', row.names=TRUE)

ciH <- confint(mH2, parm = "beta_", method = "Wald")
write.csv(as.data.frame(ciH), 'statistics/dataframes/psim_crl_H_ci.csv', row.names=TRUE)


# continuous representation
dfJ <- read.csv('analyses/dataframes/psim_crl_J_data_tr.csv')

# treat categorical variables as factors
dfJ$subject <- factor(dfJ$subject)
dfJ$exp_type <- factor(dfJ$exp_type)

mJ1 <- lmer(log_irt ~ psim + ssim + abs_lag + outpos + ncr + len_ph + exp_type + 
              psim:ssim + psim:abs_lag + ssim:abs_lag + psim:exp_type +
              (1 | subject) + (1 | session) + (1 | list), data=dfJ)

mJ2 <- lmer(log_irt ~ psim + ssim + abs_lag + outpos + ncr + len_ph + exp_type + 
              psim:ssim + psim:abs_lag + ssim:abs_lag + psim:exp_type + 
              (1 | subject) + (1 | session:exp_type) + (1 | list), data=dfJ)

mJ3 <- lmer(log_irt ~ psim + ssim + abs_lag + outpos + ncr + len_ph + exp_type + 
              psim:ssim + psim:abs_lag + ssim:abs_lag + psim:exp_type +
              (1 | subject) + (1 | session:exp_type) + (1 | list:exp_type), data=dfJ)

# model selection
aicJ <- AIC(mJ1, mJ2, mJ3)
bicJ <- BIC(mJ1, mJ2, mJ3)

# save out AIC, BIC
mJ_select <- data.frame(Model = rownames(aicJ), dof = aicJ$df, AIC = aicJ$AIC, BIC = bicJ$BIC)
write.csv(mJ_select, 'statistics/dataframes/psim_irt_J_ms.csv', row.names=FALSE)

# save out fixed effects
summaryJ <- summary(mJ2)
feJ <- summaryJ$coefficients
write.csv(as.data.frame(feJ), 'statistics/dataframes/psim_irt_J_lme.csv', row.names=TRUE)

ciJ <- confint(mJ2, parm = "beta_", method = "Wald")
write.csv(as.data.frame(ciJ), 'statistics/dataframes/psim_irt_J_ci.csv', row.names=TRUE)
