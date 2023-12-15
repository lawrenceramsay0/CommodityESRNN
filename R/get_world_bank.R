#' Gets the worldbank data from the website
#'
#' @param url1 url to world bank data
#'
#' @return data.table
#' @export
#'
#' @examples
#' See get_steel_data
get_world_bank <- function(url1 ="https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx",
                           forecast_date) {


  httr::GET(url1, httr::write_disk(tf <- tempfile(fileext = ".xls")))
  # Data/CMO-Historical-Data-Monthly.xlsx
  world_bank <- readxl::read_xlsx(
    tf,
    sheet = "Monthly Prices", skip = 6) %>%
    setnames(., old = "...1", new= "Month") %>%
    data.table() %>%
    .[, .(Month, CRUDE_BRENT, NGAS_EUR, IRON_ORE, iNATGAS, NGAS_US)] %>%
    #setnames(., old = "IRON_ORE", new = "IRON_ORE_WB") %>%
    .[, Mth := lubridate::as_date(paste(stringr::str_split_i(Month, "M", 1),stringr::str_split_i(Month, "M", 2),1,sep="-"),format = "%Y-%m-%d")] %>%
    .[, Month := NULL] %>%
    .[, iNATGAS := as.numeric(iNATGAS)]

  names(world_bank)[-match("Mth",names(world_bank))] <- paste0("WB_", names(world_bank)[-match("Mth",names(world_bank))])

  if (max(world_bank$Mth) < (lubridate::as_date(forecast_date) %m-% months(2) ))  {
    warning(paste0("World Bank Data might be out of date, please check. Latest world bank month: ", max(world_bank$Mth)))
  }

  return(world_bank)

}
