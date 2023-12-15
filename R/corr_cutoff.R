#' Wrapper for caret findCorrelation. Cuts off features with A numeric value for the
#' pair-wise absolute correlation cutoff above a certain amount.
#'
#' @param X the feature set to run cutoff on
#' @param cutoff cutoff value
#' @param verbose printing details
#'
#' @return matrix
#' @export
#'
#' @examples
#' corr_cutoff(X = matrix)
corr_cutoff <- function(X, cutoff=0.90, verbose = FALSE) {

  st_cor <- base::suppressWarnings(cor(data.frame(X)))
  # remove nas if standard deviation is 0

  st_cor <- st_cor[!rowSums(is.na(st_cor))==(nrow(st_cor)-1), !colSums(is.na(st_cor))==(ncol(st_cor)-1)]
  hc = base::suppressWarnings(caret::findCorrelation(st_cor, cutoff=cutoff, verbose = verbose)) # put any value as a "cutoff"
  hc = sort(hc)
  X_ft = X[,-c(hc)]

  return(X_ft)

}
