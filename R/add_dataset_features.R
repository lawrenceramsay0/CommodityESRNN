#' Add extra features to dataset. This is for feature expansion and modelling purposes These values will be highly correlated with
#' each other so these need to be filtered when modelled
#'
#' @param dat data table of features
#' @param tgt_ori_col_name column name that will be used as the target
#' @param derived_codes the derived codes. this is used to add extra data for the dervied forecast. Omitted if not present
#'
#' @return data.table
#' @export
#'
#' @examples
#'  st_ft <- SteelForecast::add_dataset_features(st, tgt_ori_col_name = "MB_STE_0028", derived_codes = "MB_STE_0892")
add_dataset_features <- function(dat, tgt_ori_col_name, derived_codes) {
#TODO: remove features
  dat_cols <- grepl("Mth|dte|zero_day|pub_date|tgt",names(dat))

  dat_cols <- names(dat)[!dat_cols]

  # present_derived_codes <- derived_codes[derived_codes %in% names(dat)]
  #
  # if (length(present_derived_codes) != length(derived_codes)) {
  #   warning("Not all derived codes are present in the dataset, removing")
  # }

  dat_ft <-
    dat %>%
    copy() %>%
    .[order(dte)] %>%
    # {if (length(present_derived_codes) != 0) .[, paste0("core_diff_", present_derived_codes) :=
    #                                              lapply(.SD, function(x) x-.[,get(tgt_ori_col_name)]),
    #                                            .SDcols = present_derived_codes] else .} %>%
    .[, paste0("diff_", dat_cols) := lapply(.SD, xts::diff.xts, na.pad = T),
      .SDcols = dat_cols] %>%
    .[, paste0("ravg_mt_", dat_cols) := lapply(.SD, zoo::rollmean, k = 30, na.pad = TRUE, align = "right"),
      .SDcols = dat_cols] %>%
    .[, paste0("ravg_qt_", dat_cols) := lapply(.SD, zoo::rollmean, k = 90, na.pad = TRUE, align = "right"),
      .SDcols = dat_cols] %>%
    .[, paste0("ravg_yr_", dat_cols) := lapply(.SD, zoo::rollmean, k = 365, na.pad = TRUE, align = "right"),
      .SDcols = dat_cols] %>%
    .[, paste0("lag28_", dat_cols) := lapply(.SD, shift, n = 28, type = "lag"),
      .SDcols = dat_cols] %>%
    .[, paste0("lag7_", dat_cols) := lapply(.SD, shift, n = 7, type = "lag"),
      .SDcols = dat_cols] %>%
    .[, paste0("lag1_", dat_cols) := lapply(.SD, shift, n = 1, type = "lag"),
      .SDcols = dat_cols] %>%
    .[, paste0("yoy_", dat_cols) := lapply(.SD, xts::diff.xts, lag = 365, na.pad = T),
      .SDcols = dat_cols] %>%
    .[, paste0("mom_", dat_cols) := lapply(.SD, xts::diff.xts, lag = 30, na.pad = T),
      .SDcols = dat_cols] %>%
    # {if (length(present_derived_codes) != 0) .[, paste0("diff_core_diff_", present_derived_codes) :=
    #                                              lapply(.SD, xts::diff.xts, na.pad = T),
    #                                            .SDcols = paste0("core_diff_", present_derived_codes)] else .}  %>%
    setnafill(., type = "nocb") #%>%

  return(dat_ft)

}
