#' Forward selection process
#'
#' @param train_X_all the training set
#' @param cutoff a correlation feature cutoff is done before the forward selection to prevent
#' singular matricies. This is how severe to be with the correlated valeus
#' @param nvmax max features to consider for the forward selection
#' @param features the number of features to
#' @param test_X_all the test set only gets amended if validation is true
#' @param validation are we testing the accuracy of the forecast or just running the forecast and getting the
#' best features?
#' @param train_y the target
#' @param method step method
#'
#' @return data.table
#' @export
#'
#' @examples
#' SteelForecast::create_step_selection(train_X, test_X, y, validation = TRUE)
create_step_selection <- function(train_X_all, test_X_all, train_y, validation,
                                  cutoff_val=0.90, nvmax = 20, features = nvmax,
                                  method = "forward") {

  train_X_all_ft <- corr_cutoff(train_X_all, cutoff=cutoff_val)
  if (validation) test_X_all_ft <- test_X_all[, colnames(train_X_all_ft)]

  if (ncol(train_X_all_ft) > nvmax) {

    cutoff <- cutoff_val
    boolFalse<-F
    while(boolFalse==F)
    {
      tryCatch({

        if (cutoff <= 0.2) {
          browser()
          st_ft_fwd <- leaps::regsubsets(train_y ~ ., data = data.frame(train_X_all_ft[, 1:50]), method = method, nvmax = nvmax)
        }

        print(paste0("-----cutoff: ", cutoff, " with features:", ncol(train_X_all_ft)))
        st_ft_fwd <- leaps::regsubsets(train_y ~ ., data = data.frame(train_X_all_ft), method = method, nvmax = nvmax)

        boolFalse<-T
      },error=function(e){

        print("Error at cutoff value, reducing")
        print(e$message)
        cutoff <<- cutoff-0.1

        train_X_all_ft <<- corr_cutoff(train_X_all, cutoff=cutoff)
        if (validation) test_X_all_ft <<- test_X_all[, colnames(train_X_all_ft)]

      },finally={})
    }


    st_ft_fwd_summ <- summary(st_ft_fwd)

    col <- t(st_ft_fwd_summ[["which"]])

    col_names <- rownames(col)

    feature_select_dt <- cbind(data.table(col), col_names)

    names(feature_select_dt) <- paste0("col_", names(feature_select_dt))

    #if selected row isnt avaiable then get the best option
    #the most that can be selected
    if (paste0("col_", features) %in% names(feature_select_dt)) {

      feature_select_cols <-
        feature_select_dt %>%
        .[get(paste0("col_", features)) == TRUE, col_col_names] %>%
        .[-1]

    } else {

      feature_select_cols <-
        feature_select_dt %>%
        .[get(paste0(names(feature_select_dt)[length(names(feature_select_dt))-1])) == TRUE, col_col_names] %>%
        .[-1]

    }

    #feature_select_cols_fwd <- feature_select_cols
    train_X <- train_X_all[, feature_select_cols]
    if (validation) test_X <- test_X_all[, feature_select_cols]

  } else {

    train_X <- train_X_all_ft
    if (validation) test_X <- train_X_all_ft

  }


  if (validation) {
    return(list(train_X, test_X))
  } else {
    return(train_X)
  }

}
