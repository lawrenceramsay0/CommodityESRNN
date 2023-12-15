
devtools::load_all()

dat <- data.table(readr::read_csv("data/dat.csv"))

gd <- data.table(expand.grid(ds = dat$dte, unique_id = names(dat)[-1])) %>%
  .[, x := "Price"] %>%
  .[, X_id := paste0("V", .I + 1), by = "unique_id"]


dat_esrnn <-
  dat %>%
  .[, !c("dte")] %>%
  .[, shift(.SD)] %>%
  as.matrix()

dat_esrnn[1, ] = names(dat)[-1]

dat_esrnn <- t(dat_esrnn)

readr::write_csv(gd, "data/esrnn_meta.csv")

readr::write_csv(gd, "data/esrnn_dat.csv")


