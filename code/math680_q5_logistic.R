#===== load data =====#
library(R.matlab)
dat_train <- readMat("data/Q4c_movies/moviesTrain.mat")
dat_group <- readMat("data/Q4c_movies/moviesGroups.mat")

X_train <- dat_train$trainRatings
y_train <- dat_train$trainLabels
groups <- dat_group$groupLabelsPerRating

Xc_train <- scale(X_train, scale = F)
yc_train <- scale(y_train, scale = F)
Xbar <- attributes(Xc_train)$`scaled:center`
ybar <- attributes(yc_train)$`scaled:center`

#===== functions =====#
norm_p <- function(v, p) {
  sum(abs(v)^p)^(1/p)
}
grad_g <- function(X, y, b) {
  -crossprod(X, y)
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
group_idx <- list()
group_idx[[1]] <- 1:3 # age1, age2, age3
group_idx[[2]] <- 4:6 # lwt1, lwt2, lwt3
group_idx[[3]] <- 7:8 # white, black
group_idx[[4]] <- 9 # smoke
group_idx[[5]] <- 10:11 # ptl1, ptl2m
group_idx[[6]] <- 12 # ht
group_idx[[7]] <- 13 # ui
group_idx[[8]] <- 14:16 # ftv1, ftv2, ftv3m
n_groups <- length(group_idx)

w <- sapply(group_idx, function(groupj) sqrt(length(groupj)))
beta_init <- rep(0, ncol(Xc))

beta <- matrix(nrow = max_steps, ncol = length(beta_init))
beta[1, ] <- beta_init - t_step * grad_g(Xc, yc, beta_init)

for (k in 2:max_steps) {
  # update step
  beta[k,] <- beta[k - 1,] - t_step * grad_g(Xc, yc, beta[k - 1,])
  
  # proximal step
  for (j in 1:n_groups) {
    beta[k, group_idx[[j]]] <- 
      Stilde_groupj(beta[k,  group_idx[[j]]], lambda, t_step, w[j])
  }
}

f <- apply(beta, 1, function(b) {
  h <- lambda * sum(w * sapply(group_idx, function(groupj) norm_p(b[groupj], 2)))
  crossprod(yc - Xc %*% b) + h
})


plot(f - fstar, log = 'xy', type = 'l')

