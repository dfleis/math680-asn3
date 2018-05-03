#===== load data =====#
library(R.matlab)
dat_train <- readMat("data/Q4c_movies/moviesTrain.mat")
dat_group <- readMat("data/Q4c_movies/moviesGroups.mat")

X_train <- dat_train$trainRatings
y_train <- as.numeric(dat_train$trainLabels)
groups <- dat_group$groupLabelsPerRating

Xc_train <- scale(X_train, scale = F)
yc_train <- as.numeric(scale(y_train, scale = F))
Xbar <- attributes(Xc_train)$`scaled:center`
ybar <- attributes(yc_train)$`scaled:center`

#===== functions =====#
norm_p <- function(v, p) {
  sum(abs(v)^p)^(1/p)
}
grad_g <- function(X, y, b) {
  expXb <- exp(X %*% b)
  -t(y %*% X) + crossprod(X, expXb/(1 + expXb))
}
Stilde_groupj <- function(beta_groupj, lambda, t_step, w_groupj) {
  beta_groupj_norm2 <- norm_p(beta_groupj, 2)
  
  beta_groupj/beta_groupj_norm2 * 
    max(beta_groupj_norm2 - lambda * t_step * w_groupj, 0)  
}

#===== set parameters =====#
fstar <- 336.207
lambda <- 5
t_step <- 1e-4
max_steps <- 1e3
w <- sqrt(table(groups))
n_groups <- length(w)

beta_init <- rep(0, ncol(Xc_train))

beta <- matrix(nrow = max_steps, ncol = length(beta_init))
beta[1, ] <- beta_init - t_step * grad_g(Xc_train, yc_train, beta_init)

for (k in 2:max_steps) {
  # update step
  beta[k,] <- beta[k - 1,] - t_step * grad_g(Xc_train, yc_train, beta[k - 1,])
  
  # proximal step
  for (j in 1:n_groups) {
    group_idx <- which(groups == j)
    
    beta[k, group_idx] <- 
      Stilde_groupj(beta[k, group_idx], lambda, t_step, w[j])
  }
}

f <- apply(beta, 1, function(b) {
  for (j in 1:n_groups) {
    group_idx <- which(groups == j)
    norm_p(b[group_idx], 2)
  }
  
  
  h <- lambda * sum(w * sapply(group_idx, function(groupj) norm_p(b[groupj], 2)))
  crossprod(yc - Xc_train %*% b) + h
})


plot(f - fstar, log = 'xy', type = 'l')

