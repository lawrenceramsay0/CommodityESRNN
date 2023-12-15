#' Pulls construction data from euro stat
#'
#' @return data.table
#' @export
#'
#' @examples
#' get_construction_eurostat()
get_construction_eurostat <- function() {

  #https://ec.europa.eu/eurostat/databrowser/view/EI_ISBU_M__custom_5618662/default/table?lang=en

  #https://ec.europa.eu/eurostat/databrowser/view/EI_ISBU_M__custom_5645058/settings_1/table?lang=en

  constr <- data.table::fread("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/EI_ISBU_M/M.I2015.NSA+SCA.IS-IP.F+B-D_F+F_CC1+F_CC2+F_CC11_X_CC113.DE/?format=SDMX-CSV&startPeriod=1980-01") %>%
    .[s_adj == "NSA"] %>%
    .[, .(nace_r2, TIME_PERIOD, OBS_VALUE)] %>%
    setnames(., old = "nace_r2", new = "indicator") %>%
    setnames(., old = "TIME_PERIOD", new = "Mth") %>%
    dcast(., Mth ~ indicator, value.var = "OBS_VALUE") %>%
    .[, Mth := lubridate::as_date(paste0(Mth, "-01"), format = "%Y-%m-%d")]

  names(constr)[-1] <- paste0("CON_DE_",names(constr)[-1])

  return(constr)

}
