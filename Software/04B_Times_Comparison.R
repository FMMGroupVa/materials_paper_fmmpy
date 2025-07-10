

source("3DFMM/auxMultiFMM.R") 

df <- read.csv("Data/ECG_data.csv", header = FALSE)
exampleData <- t(df[, 350:850])  # Keep only columns 350:850

# Function to run the estimation
runFMM <- function(data, nBack = 5, maxIter = 8) {
  fitMultiFMM(vDataMatrix = data, nBack = nBack, maxIter = maxIter, parallelize = F)
}

# Number of repeats
N_REPEATS <- 100

# Vectors to store times
times <- numeric(N_REPEATS)

# Repeated runs
for (i in 1:N_REPEATS) {
  print(i)
  start_time <- Sys.time()
  res <- runFMM(exampleData, nBack = 5, maxIter = 8)
  end_time <- Sys.time()
  times[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

# Stats
mean_time <- mean(times)
max_time <- max(times)

cat(sprintf("\nFMM in R - Mean execution time over %d runs: %.4f s\n", N_REPEATS, mean_time))
cat(sprintf("FMM in R - Max time: %.4f s\n", max_time))

# Write summary to text file
summary_text <- sprintf(
  "FMM in R:\nMean execution time over %d runs: %.4f s\nMax time: %.4f s\n",
  N_REPEATS, mean_time, max_time
)

writeLines(summary_text, "Results/execution_times_R_summary.txt")

cat("\nResults saved to 'execution_times_R_summary.txt'\n")
