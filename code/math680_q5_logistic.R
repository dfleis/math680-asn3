#===== load data =====#
library(R.matlab)
dat_train <- readMat("data/Q4c_movies/moviesTrain.mat")
dat_group <- readMat("data/Q4c_movies/moviesGroups.mat")
dat_test <- readMat("data/Q4c_movies/moviesTest.mat")

X_train <- dat_train$trainRatings
y_train <- as.numeric(dat_train$trainLabels)
X_test <- dat_test$testRatings
y_test <- as.numeric(dat_test$testLabels)
groups <- as.numeric(dat_group$groupLabelsPerRating)

#===== functions =====#
norm_p <- function(v, p) {
  sum(abs(v)^p)^(1/p)
}
grad_g <- function(X, y, b) { # logistic gradient
  Xb <-  X %*% b
  logist <- ifelse(Xb > 500, 1, exp(Xb)/(1 + exp(Xb)))
  
  grad_out <- -t(X) %*% (y - logist)
  grad_out
}
h_fun <- function(b, groups, lambda) {
  w <- sqrt(tabulate(groups))
  
  h_out <- 0
  for (j in 1:length(w)) {
    group_idx <- groups == j
    h_out <- h_out + w[j] * norm_p(b[group_idx], 2)
  }
  lambda * h_out
}
g_fun <- function(X, y, b) {
  Xb <- X %*% b
  term1 <- -as.numeric(y %*% Xb)
  term2 <- sum(ifelse(Xb > 500, Xb, log(1 + exp(Xb))))
  term1 + term2
}
f_obj <- function(X, y, b, groups, lambda) {
  g_fun(X, y, b) + h_fun(b, groups, lambda)
}
Stilde_groupj <- function(beta_groupj, lambda, t_step, w_groupj) {
  beta_groupj_norm2 <- norm_p(beta_groupj, 2)
  
  beta_groupj/beta_groupj_norm2 * 
    max(beta_groupj_norm2 - lambda * t_step * w_groupj, 0)  
}
prox <- function(x, groups, lambda, t_step) {
  w <- sqrt(tabulate(groups))
  n_groups <- length(w)
  
  x_out <- rep(NA, length(x))
  for (j in 1:n_groups) {
    group_idx <- which(groups == j)
    
    x_out[group_idx] <- 
      Stilde_groupj(x[group_idx], lambda, t_step, w[j])
  }
  x_out
}

#===== set parameters =====#
fstar <- 336.207
lambda <- 5
t_step <- 1e-4
max_steps <- 1e2

w <- sqrt(tabulate(groups))
n_groups <- length(w)

X <- X_train
y <- y_train

#===== proximal GD =====#
beta_init <- rep(0, ncol(X))
beta_prox <- matrix(nrow = max_steps, ncol = length(beta_init))
beta_prox[1, ] <- beta_init - t_step * grad_g(X, y, beta_init)

for (k in 2:max_steps) {
  prox_arg <- beta_prox[k - 1,] - t_step * grad_g(X, y, beta_prox[k - 1,])
  beta_prox[k,] <- prox(prox_arg, groups, lambda, t_step)
}
#===== accelerated proximal GD =====#
beta_init_m1 <- rep(0, ncol(X))
beta_init_00 <- rep(0, ncol(X))

beta_acc <- matrix(nrow = max_steps + 2, ncol = ncol(X))
beta_acc[1, ] <- beta_init_m1
beta_acc[2, ] <- beta_init_00 

for (k in 3:nrow(beta_acc)) {
  # momentum step
  v <- beta_acc[k - 1,] + (k - 4)/(k - 1) * (beta_acc[k - 1,] - beta_acc[k - 2,])
  
  # proximal step
  prox_arg <- v - t_step * grad_g(X, y, beta_acc[k - 1,])
  beta_acc[k,] <- prox(prox_arg, groups, lambda, t_step)
}

#===== backtracking GD =====#
t_step_init <- 1
beta_shrink <- 0.1

beta_init <- rep(0, ncol(X))
beta_back <- matrix(nrow = max_steps + 1, ncol = length(beta_init))
beta_back[1, ] <- beta_init

for (k in 2:nrow(beta_back)) {
  t_step <- t_step_init
  
  grad_g_val <- grad_g(X, y, beta_back[k - 1,])
  g_val <- g_fun(X, y, beta_back[k - 1,])
  prox_arg <- beta_back[k - 1,] - t_step * grad_g_val
  Gt <- 1/t_step * (beta_back[k - 1,] - prox(prox_arg, groups, lambda, t_step))
  
  LHS <- g_fun(X, y, beta_back[k - 1,] - t_step * Gt)
  RHS <- g_val - t_step * as.numeric(crossprod(grad_g_val, Gt)) + 
    t_step/2 * norm_p(Gt, 2)^2
  
  while (LHS > RHS) {
    t_step <- t_step * beta_shrink
    
    grad_g_val <- grad_g(X, y, beta_back[k - 1,])
    g_val <- g_fun(X, y, beta_back[k - 1,])
    prox_arg <- beta_back[k - 1,] - t_step * grad_g_val
    Gt <- 1/t_step * (beta_back[k - 1,] - prox(prox_arg, groups, lambda, t_step))
    
    LHS <- g_fun(X, y, beta_back[k - 1,] - t_step * Gt)
    RHS <- g_val - t_step * as.numeric(crossprod(grad_g_val, Gt)) + 
      t_step/2 * norm_p(Gt, 2)^2
  }
  
  beta_back[k,] <- prox(prox_arg, groups, lambda, t_step)
}

#===== compute objective values =====#
f_prox <- apply(beta_prox, 1, function(b) {
  f_obj(X, y, b, groups, lambda)
})
f_acc <- apply(beta_acc, 1, function(b) {
  f_obj(X, y, b, groups, lambda)
})
f_back <- apply(beta_back, 1, function(b) {
  f_obj(X, y, b, groups, lambda)
})

plot(f_prox - fstar, 
     ylim = range(c(f_prox, f_acc, f_back) - fstar), 
     xlab = "Step", ylab = "f - fstar",
     log = 'xy', type = 'l', lwd = 2, lty = "dotdash")
lines(f_acc - fstar, col = 'red', lwd = 2, lty = "dashed") 
lines(f_back - fstar, col = 'blue', lwd = 2, lty = "solid")
legend("topright", legend = c("Prox.", "Acc. Prox.", "Back. Prox."),
       lwd = 2, seg.len = 1.5, 
       col = c("black", "red", "blue"),
       lty = c("dotdash", "dashed", "solid"))




#===== use acc prox GD to fit test data =====#
acc_min_idx <- which(f_acc == min(f_acc))[1]
beta_hat <- beta_acc[acc_min_idx,]

# compute fitted probabilities
pi_hat <- exp(X_test %*% beta_hat)/(1 + exp(X_test %*% beta_hat))
# compute fitted classes
yhat <- ifelse(pi_hat > 0.5, 1, 0)
table(y_test, yhat)

# under 40 -> y = 1
group_is_important <- rep(NA, max(groups))
for (j in 1:length(group_is_important)) {
  group_idx <- which(groups == j)
  group_is_important[j] <- sum(abs(beta_hat[group_idx])) != 0
}
unlist(dat_group$groupTitles)[group_is_important]

sapply(1:max(groups), function(j) {
  which
})




