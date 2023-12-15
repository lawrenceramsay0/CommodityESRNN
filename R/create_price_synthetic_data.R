create_price_synthetic_data <- function(prc_daily, plot = F, verbose = T, exclude_cols = c("Mth", "dte", "EUR_USD")) {

  prc_daily_synth <- copy(prc_daily)
  plot_list <- list()
  plot_data <- data.table()
  browser()
  if (verbose) print("Creating synthetic data for:")
  #col = names(prc_daily)[!names(prc_daily) %in% exclude_cols][1]
  for (col in names(prc_daily)[!names(prc_daily) %in% exclude_cols]) {

    na_length <- length(prc_daily[is.na(get(col)), get(col)])

    if (na_length > 0) {

      if (verbose) print(col)
      exclude_cols_loop <- c(exclude_cols, col)

      dt <- prc_daily_synth[order(-dte)]

      dte_no_na <- dt[!is.na(get(col)), get("dte")]
      dte_na <- dt[is.na(get(col)), get("dte")]

      X_reg <- as.matrix(dt[is.na(get(col)), !..exclude_cols_loop]) %>%
        .[, colSums(is.na(.)) == 0]

      X <- as.matrix(dt[!is.na(get(col)), !..exclude_cols_loop]) %>%
        .[, colnames(X_reg)]

      y <- dt[!is.na(get(col)),get(col)]

      mdl <- forecast::auto.arima(y, xreg = X)

      fcst <- forecast::forecast(mdl, h = na_length, xreg = X_reg)

      prc_daily_synth <- dt[is.na(get(col)), (col) := fcst$mean]

      if (plot) {
        plot_data <- rbind(plot_data, data.table(symbol = col ,
                                                 value = c(y, fcst$mean),
                                                 dte = c(dte_no_na, dte_na),
                                                 data_type = c(rep("real", length(y)), rep("synthetic", length(fcst$mean)))
                                                  ))
      }

    }

  }
  if (plot) {
    return(list(plot_data, prc_daily_synth))
  } else {
    return(prc_daily_synth)
  }


}
