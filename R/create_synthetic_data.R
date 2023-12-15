#' Creates synthetic data for any data table that has na values contained
#'
#' Works by forecasting backwards fort he nas using regressers if possible
#' method of dealing with nas
#' features selected with forward selection
#' Uses and unsupervised method of forecasting so doesn't check how accurate it is
#' It is only to guess some values for the variation and so the feature could be used in the forecast
#' The testing of the feature importance happens in the main model selection
#'
#' @param dat data table of data
#' @param plot whether to add the plot data to the return ouput
#' @param verbose printing values
#' @param exclude_cols which columns to exclude
#' @param cutoff_val cutoff vlaue for the forward selection
#' @param nvmax the max of the forward selection
#' @param dte_col the column for the date values in the timeseries
#' @param y_only if true then only usees the y value for the forecast with no regressors. Use this
#' for the sake of speed
#'
#' @return data.table
#' @export
#'
#' @examples
#' ceic_synth_list <- create_synthetic_data(ceic, plot = T, verbose = verbose, dte_col = "Mth")
#' ceic_synth <- ceic_synth_list[[2]]
#' ceic_synth_plot <- ceic_synth_list[[1]]
create_synthetic_data <- function(dat, plot = F, verbose = T, exclude_cols = NULL,
                                  dte_col, cutoff_val = 0.8, nvmax = 10, y_only = FALSE) {

  plot_list <- list()
  plot_data <- data.table()

  dat <-
    dat %>%
    .[order(-get(dte_col))] %>%
    .[, ind := .I]

  #Remove columns from the data if they are excluded
  if (!is.null(exclude_cols)) {
    for (ex_col in exclude_cols) {

      if (ex_col %in% names(dat)) {
        dat <- dat[, !..ex_col]
      }

    }
  }

  dat_synth <- copy(dat)

  indexes <- c(dte_col, "ind", exclude_cols)

  if (verbose) print("Creating synthetic data for:")

  for (col in names(dat)[!names(dat) %in% c(exclude_cols, dte_col)]) {

    na_length <- length(dat[is.na(get(col)), get(col)])

    if (na_length > 0) {

      if (verbose) print(paste0(col, " NAs: ", na_length))

      ind_no_na <- dat[!is.na(get(col)), ind]
      ind_na <- dat[is.na(get(col)), ind]

      dte_no_na <- dat[ind_no_na, get(dte_col)]
      dte_na <- dat[ind_na, get(dte_col)]

      y <- dat[ind_no_na,get(col)]

      if (y_only) {

        mdl <- forecast::auto.arima(y, stepwise = F)
        fcst <<- forecast::forecast(mdl, h = na_length, level = 90)

      } else {

        X <- dat %>%
          copy() %>%
          .[ind_no_na, !..indexes] %>%
          as.matrix(.) %>%
          .[, colSums(is.na(.)) == 0] %>% #columns that dont have nas
          # Select cols via forward selection
          Thesis::create_step_selection(train_X_all = ., train_y = y, validation = FALSE,
                                                 nvmax = nvmax, cutoff = cutoff_val)

        if (ncol(X) == 0 | na_length == 1) {

          mdl <- forecast::auto.arima(y, stepwise = F)
          fcst <<- forecast::forecast(mdl, h = na_length, level = 90)

        } else {

          #Creates regression set from na of column
          X_reg <- dat %>%
            copy() %>%
            .[ind_na, !..dte_col] %>%
            as.matrix(.) %>%
            .[, colnames(X)] %>%
            .[, colSums(is.na(.)) == 0]

          #Returns to the original and take anything away that is na in the X regression
          X <- X[, colnames(X_reg)]

          #Catch singular forecasts from regressors
          tryCatch({

            mdl <- forecast::auto.arima(y, xreg = X, stepwise = F)
            fcst <<- forecast::forecast(mdl, h = na_length, level = 90,
                                        xreg = X_reg)


          },error=function(e){

            mdl <- forecast::auto.arima(y, stepwise = F)
            fcst <<- forecast::forecast(mdl, h = na_length, level = 90)

          },finally={})

        }
      }

      dat_synth <- dat[is.na(get(col)), (col) := fcst$mean]

      if (plot) {
        plot_data <- rbind(plot_data, data.table(symbol = col ,
                                                 value = c(y, fcst$mean),
                                                 dte = c(dte_no_na, dte_na),
                                                 data_type = c(rep("real", length(y)), rep("synthetic", length(fcst$mean)))))
      }

    }

  }

  if ("ind" %in% names(dat_synth)) {
    dat_synth[, ind := NULL]
  }

  if (plot) {
    return(list(plot_data, dat_synth))
  } else {
    return(dat_synth)
  }


}
