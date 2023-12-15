#' Gets sentiment data from Eurostat
#'
#' @return dasta.table
#' @export
#'
#' @examples
#' see get_steel_data
get_sentiment_eurostat <- function() {

  #https://ec.europa.eu/eurostat/databrowser/view/EI_BSSI_M_R2__custom_3882809/default/table?lang=en

  sentiment <- data.table::fread("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/EI_BSSI_M_R2/M.BS-CCI-BAL+BS-ESI-I+BS-ICI-BAL+BS-RCI-BAL+BS-CSMCI-BAL+BS-SCI-BAL.NSA.DE/?format=SDMX-CSV&startPeriod=1980-01") %>%
    .[s_adj == "NSA"] %>%
    .[, .(indic, TIME_PERIOD, OBS_VALUE)] %>%
    setnames(., old = "TIME_PERIOD", new = "Mth") %>%
    dcast(., Mth ~ indic, value.var = "OBS_VALUE") %>%
    .[, Mth := lubridate::as_date(paste0(Mth, "-01"), format = "%Y-%m-%d")]

  return(sentiment)

}
