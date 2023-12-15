
devtools::load_all()

dat <- data.table(readr::read_csv("data/dat.csv"))

 #%>%
  #.[, X_id := paste0("V", seq_len(.N) + 1), by = c("unique_id")]
#

# dat_esrnn <-
#   dat %>%
#   .[, !c("dte")] %>%
#   .[, shift(.SD)] %>%
#   as.matrix()

y_df <-
  dat %>%
  melt(., id.vars = "dte", variable.name = "unique_id", value.name = "y") %>%
  setnames(., old = "dte", new = "ds") %>%
  .[, y_hat_naive2 := shift(y), by = "unique_id"] %>%
  na.omit()

X_df <- data.table(expand.grid(ds = dat$dte, unique_id = names(dat)[-1])) %>%
  .[, x := "Price"] %>%
  .[ds %chin% unique(y_df[, ds])]

#dat_esrnn[1, ] = names(dat)[-1]

#dat_esrnn <- data.table(t(dat_esrnn))

readr::write_csv(X_df, "data/X_df.csv")

readr::write_csv(y_df, "data/y_df.csv")


