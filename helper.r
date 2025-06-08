library("ggplot2")

"The specification of the drift term µ is unimportant because
it will not affect option prices."

random_heston <- function(n, s0, v0, dt, theta, k, sigma, rho) {
  "
  If dt is too big, vt may go below 0 and make the simulations return NA.
  dt = 1 is a tested example that cause this.
  "
  r <- 0.02
  cov_matrix <- matrix(c(1, rho, rho, 1), nrow = 2, ncol = 2, byrow = TRUE)
  correlated_processes <- MASS::mvrnorm(n, rep(0, 2), cov_matrix)
  heston_brownian <- correlated_processes[, 1]
  volatily_brownian <- correlated_processes[, 2]

  v <- numeric(n + 1)
  s <- numeric(n + 1)
  v[1] <- v0
  s[1] <- s0
  for (i in 1:n) {
    s_vol <- sqrt(v[i]) * dt * heston_brownian[i]
    s_drift <- (r - v[i] / 2) * dt
    s[i + 1] <- s[i] * exp(s_drift + s_vol)

    v_vol <- sigma * dt * sqrt(v[i]) * volatily_brownian[i]
    v_drift <- k * (theta - v[i])
    v[i + 1] <- v[i] + v_drift + v_vol
  }
  return(s)
  # k * (theta - v0) * dt + sigma * sqrt(v0) * volatily_brownian
}

random_heston(252, 10.83, 0.1, 0.01, 0.1, 2, 0.2, 0.95)