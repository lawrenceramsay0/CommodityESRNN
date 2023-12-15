#' get_hicp_eurostat
#'
#' @return data.table
#' @export
#'
#' @examples
#' see get_steel_data
get_hicp_eurostat <- function() {

  hicp <- data.table::fread("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/PRC_HICP_MIDX/M.I05.IGD+NRG+ELC_GAS+TOT_X_FOOD_S+TOT_X_ALC_TBC+TOT_X_TBC+TOT_X_NRG+TOT_X_NRG_FOOD+TOT_X_NRG_FOOD_NP+TOT_X_NRG_FOOD_S+TOT_X_FUEL+TOT_X_HOUS+TOT_X_FROOPP.EU/?format=SDMX-CSV&startPeriod=2001-01") %>%
    .[, .(coicop, TIME_PERIOD, OBS_VALUE)] %>%
    setnames(., old = "coicop", new = "indicator") %>%
    setnames(., old = "TIME_PERIOD", new = "Mth") %>%
    dcast(., Mth ~ indicator, value.var = "OBS_VALUE") %>%
    .[, Mth := lubridate::as_date(paste0(Mth, "-01"), format = "%Y-%m-%d")]

  return(hicp)

}
