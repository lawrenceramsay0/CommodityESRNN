
devtools::load_all()

path <- "C:\\Users\\lawre\\OneDrive - City, University of London\\Thesis\\archive\\"

headlines <- readr::read_csv(paste0(path, "raw_partner_headlines.csv")) %>%
  data.table()

analyst <- readr::read_csv(paste0(path, "raw_analyst_ratings.csv")) %>%
  data.table()

iron_companies <- c("BHP", "VALE", "CLF", "X", "MSB",
                    "RIO", "XME", "MALRF", "SXC", "STLD",
                    "NUE", "HRGLF")

companies = data.table(company = c("BHP", "VALE", "CLF", "X", "MSB", "RIO",
                                             "XME", "MALRF", "SXC", "STLD", "NUE", "HRGLF",
                                             "XOM","SHEL", "TTE", "CVX", "BP", "MPC", "VLO"),
  sector = c("i", "i", "i", "i","i", "i", "i", "i","i", "i", "i", "i", "c",
             "c", "c", "c", "c", "c", "c"))


headlines <- headlines[stock %chin% companies[, company], .(headline, date, stock)]
analyst <- analyst[stock %chin% companies[, company], .(headline, date, stock)]

headlines_iron <- headlines[headline %like% "iron", .(headline, date, stock)]
analyst_iron <- analyst[headline %like% "iron", .(headline, date, stock)]

headlines_crd <- headlines[headline %like% "crude", .(headline, date, stock)]
analyst_crd <- analyst[headline %like% "crude", .(headline, date, stock)]

news_arc <- unique(rbind(headlines, analyst, headlines_iron, analyst_iron, headlines_crd, analyst_crd)) %>%
  merge(., companies, by.y = "company", by.x = "stock") %>%
  .[, dte := as_date(date)]

cnt <- news_arc[, .N, by = c("dte")]

companies <- sort(unique(c(unique(headlines[, stock]),  unique(analyst$stock))))

max(iron$date)

readr::write_csv(news_arc, paste0("data/news_arc.csv"))

