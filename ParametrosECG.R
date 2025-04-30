
library(readxl)


cols <- read_excel("C:/Users/Christian/Desktop/completeMultiLeadResults.xlsx")
sub <- cols[,c(5,12:21)]

percentiles <- apply(sub[sub$Class=="NORM",2:11], 2, quantile, probs = c(0.01, 0.025, 0.975, 0.99), na.rm = TRUE)
percentiles <- apply(sub[,2:11], 2, quantile, probs = c(0.01, 0.025, 0.975, 0.99), na.rm = TRUE)
hist(sub$OmegaP)
