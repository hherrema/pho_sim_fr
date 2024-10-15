# imports
library(lme4)
library(lmerTest)

# clustering and recall probability
df <- read.csv('analyses/dataframes/cl_pr.csv')

# treat categorical variables as factors
df$subject <- factor(df$subject)
df$exp_type <- factor(df$exp_type)

# binary metric
mH1 <- lmer(p_recall ~ pcs_H + scs + tcs + exp_type + l_length +
              pcs_H:scs + pcs_H:tcs + scs:tcs +
              exp_type:pcs_H + exp_type:scs + exp_type:tcs +
              (1 | subject) + (1 | session) + (1 | list), data=df)

mH2 <- lmer(p_recall ~ pcs_H + scs + tcs + exp_type + l_length + 
              pcs_H:scs + pcs_H:tcs + scs:tcs +
              exp_type:pcs_H + exp_type:scs + exp_type:tcs +
              (1 | subject) + (1 | session:exp_type) + (1 | list), data=df)

mH3 <- lmer(p_recall ~ pcs_H + scs + tcs + exp_type + l_length +
              pcs_H:scs + pcs_H:tcs + scs:tcs +
              exp_type:pcs_H + exp_type:scs + exp_type:tcs +
              (1 | subject) + (1 | session:exp_type) + (1 | list:exp_type), data=df)

# model selection
aicH <- AIC(mH1, mH2, mH3)
bicH <- BIC(mH1, mH2, mH3)

# save out AIC, BIC
mH_select <- data.frame(Model = rownames(aicH), dof = aicH$df, AIC = aicH$AIC, BIC = bicH$BIC)
write.csv(mH_select, 'statistics/dataframes/cl_pr_H_ms.csv', row.names=FALSE)

# save out fixed effects
summaryH <- summary(mH2)
feH <- summaryH$coefficients
write.csv(as.data.frame(feH), 'statistics/dataframes/cl_pr_H_lme.csv', row.names=TRUE)

ciH <- confint(mH2, parm = "beta_", method = "Wald")
write.csv(as.data.frame(ciH), 'statistics/dataframes/cl_pr_H_ci.csv', row.names=TRUE)


# continuous metric
mJ1 <- lmer(p_recall ~ pcs_J + scs + tcs + exp_type + l_length +
              pcs_J:scs + pcs_J:tcs + scs:tcs +
              exp_type:pcs_J + exp_type:scs + exp_type:tcs +
              (1 | subject) + (1 | session) + (1 | list), data=df)

mJ2 <- lmer(p_recall ~ pcs_J + scs + tcs + exp_type + l_length +
              pcs_J:scs + pcs_J:tcs + scs:tcs +
              exp_type:pcs_J + exp_type:scs + exp_type:tcs +
              (1 | subject) + (1 | session:exp_type) + (1 | list), data=df)

mJ3 <- lmer(p_recall ~ pcs_J + scs + tcs + exp_type + l_length +
              pcs_J:scs + pcs_J:tcs + scs:tcs +
              exp_type:pcs_J + exp_type:scs + exp_type:tcs +
              (1 | subject) + (1 | session:exp_type) + (1 | list:exp_type), data=df)

# model selection
aicJ <- AIC(mJ1, mJ2, mJ3)
bicJ <- BIC(mJ1, mJ2, mJ3)

# save out AIC, BIC
mJ_select <- data.frame(Model = rownames(aicJ), dof = aicJ$df, AIC = aicJ$AIC, BIC = bicJ$BIC)
write.csv(mJ_select, 'statistics/dataframes/cl_pr_J_ms.csv', row.names=FALSE)

# save out fixed effects
summaryJ <- summary(mJ2)
feJ <- summaryJ$coefficients
write.csv(as.data.frame(feJ), 'statistics/dataframes/cl_pr_J_lme.csv', row.names=TRUE)

ciJ <- confint(mJ2, parm = "beta_", method = "Wald")
write.csv(as.data.frame(ciJ), 'statistics/dataframes/cl_pr_J_ci.csv', row.names=TRUE)