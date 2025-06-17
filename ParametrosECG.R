
library(readxl)
library(ggplot2)

cols <- read_excel("C:/Users/Christian/Desktop/completeMultiLeadResults.xlsx")
sub <- cols[,c(5,10,12:21)]

percentiles <- round(apply(sub[sub$Class=="NORM" & sub$Likelihood==100,3:12], 2, quantile, probs = c(0.01, 0.025,0.1, 0.90, 0.975, 0.99), na.rm = TRUE), 2)
percentiles
percentiles <- round(apply(sub[,2:11], 2, quantile, probs = c(0.01, 0.025, 0.1, 0.90, 0.975, 0.99), na.rm = TRUE), 2)
percentiles
hist(sub$OmegaP)


ggplot(sub, aes(x = sub$AlphaP, y = sub$OmegaP)) +
  geom_density_2d_filled(alpha = 0.7) + 
  theme_minimal()
