
devtools::load_all()

options(scipen = 999)

constr <- Thesis::get_construction_eurostat() %>%
  set_colnames(c("Mth", paste0("ctr_" ,names(.)[-1])))

hicp <- Thesis::get_hicp_eurostat() %>%
  set_colnames(c("Mth", paste0("hicp_" ,names(.)[-1])))

sent <- Thesis::get_sentiment_eurostat()  %>%
  set_colnames(c("Mth", paste0("snt_" ,names(.)[-1])))

wb <- Thesis::get_world_bank(forecast_date = Sys.Date())

gt <- readr::read_csv("data/GoogleTrends.csv", skip = 2) %>%
  data.table()%>%
  .[, Month := lubridate::ymd(paste0(Month, "-01"))] %>%
  set_colnames(c("Mth", "googtnd_iron_ore", "googtnd_wti", "googtnd_inf", "googtnd_brent"))

macro <-
  wb %>%
  merge(., constr, by = "Mth", all.x = T) %>%
  merge(., hicp, by = "Mth", all.x = T) %>%
  merge(., sent, by = "Mth", all.x = T) %>%
  merge(., gt, by = "Mth", all.x = T)

prc <- data.table(readr::read_csv("data/prc.csv"))

iron_ore <- readr::read_csv("data/Iron ore fines 62% Fe CFR Futures Historical Data.csv") %>%
  data.table() %>%
  .[, Date := lubridate::dmy(Date)] %>%
  .[, .(Date, close_irn = Price, high_irn = High, low_irn = Low)] %>%
  .[, "tgt_irn" := shift(close_irn, type = "lead")]

#crd_olhc <- quantmod::as.quantmod.OHLC()

dat <-
  merge(prc, macro, by.x = "Date", by.y = "Mth", all.x = T) %>%
  merge(., iron_ore, by = "Date", all.x = T) %>%
  setnafill(., type = "locf") %>%
  .[, "tgt_crd" := shift(Close_crd, type = "lead")] %>%
  setnames(., old = "Date", new = "dte") %>%
  .[!is.na(tgt_irn)] %>%
  .[!is.na(tgt_crd)] %>%
  .[, rsi_crd := TTR::RSI(Close_crd)] %>%
  .[, momentum_crd := TTR::momentum(Close_crd)] %>%
  cbind(., set_colnames(TTR::MACD(.$Close_crd),c("macd_crd", "macd_signal_crd"))) %>%
  cbind(., set_colnames(TTR::BBands(.$Close_crd),c("bb_dn_crd", "bb_mavg_crd", "bb_up_crd", "bb_pctB_crd"))) %>%
  .[, rsi_irn := TTR::RSI(close_irn)] %>%
  .[, momentum_irn := TTR::momentum(close_irn)] %>%
  cbind(., set_colnames(TTR::MACD(.$close_irn),c("macd_irn", "macd_signal_irn"))) %>%
  cbind(., set_colnames(TTR::BBands(.$close_irn),c("bb_dn_irn", "bb_mavg_irn", "bb_up_irn", "bb_pctB_irn"))) %>%
  #Divide volumes by 1m to make the normalisation easier
  .[, Volume_sp := Volume_sp / 1000000] %>%
  .[, Volume_ns := Volume_ns / 1000000] %>%
  .[, Volume_crd := Volume_crd / 1000] %>%
  set_colnames(stringr::str_replace_all(names(.),"-", "_")) %>%
  set_colnames(stringr::str_replace_all(names(.)," ", "_")) %>%
  set_colnames(tolower(names(.)))

news_eod_raw <- readr::read_csv("data/news_eod_sentiment.csv") %>%
  data.table() %>%
  .[lbl == 0, lbl_std := -1] %>%
  .[lbl == 2, lbl_std := 0] %>%
  .[lbl == 1, lbl_std := 1]

news_arc_raw <- readr::read_csv("data/news_arc_sentiment.csv") %>%
  data.table() %>%
  .[sentiment_ctr == "neutral", lbl := 0] %>%
  .[sentiment_ctr == "positive", lbl := 1] %>%
  .[sentiment_ctr == "negative", lbl := -1]

#Combined
news_eod <-
  news_eod_raw %>%
  .[, .(fse_pol_sum = sum(polarity), fse_neg_sum = sum(neg), fse_neu_sum = sum(neu),
        fse_pos_sum = sum(pos), fse_lbl_sum_eod = sum(lbl_std),
        fse_pol_med = median(polarity), fse_neg_med = median(neg), fse_neu_med = median(neu),
        fse_pos_med = median(pos), fse_lbl_med_eod = median(lbl_std)), by = c("dte")]

news_arc <-
  news_arc_raw  %>%
  .[, .(fsa_lbl_sum_arc = sum(lbl), fsa_lbl_med_arc = median(lbl),
        fsa_sent_sum = sum(sentiment_probability_ctr),
        fsa_sent_med = median(sentiment_probability_ctr)), by = c("dte")]

news_comb <- rbind(news_arc_raw[, .(dte, lbl, sector)], news_eod_raw[, .(dte, lbl, sector)]) %>%
  .[lbl != 0] %>%
  .[, .(fsc_lbl_sum = sum(lbl), fsc_lbl_med = median(lbl)), by = c("dte")]

#By Sector
news_eod_sec <-
  news_eod_raw %>%
  .[, .(fse_pol_sum = sum(polarity), fse_neg_sum = sum(neg), fse_neu_sum = sum(neu),
        fse_pos_sum = sum(pos), fse_lbl_sum_eod = sum(lbl_std),
        fse_pol_med = median(polarity), fse_neg_med = median(neg), fse_neu_med = median(neu),
        fse_pos_med = median(pos), fse_lbl_med_eod = median(lbl_std)), by = c("dte", "sector")] %>%
  dcast(dte ~ sector, value.var = names(.)[-1][-1])

news_arc_sec <-
  news_arc_raw  %>%
  .[, .(fsa_lbl_sum_arc = sum(lbl), fsa_lbl_med_arc = median(lbl),
        fsa_sent_sum = sum(sentiment_probability_ctr),
        fsa_sent_med = median(sentiment_probability_ctr)), by = c("dte", "sector")] %>%
  dcast(dte ~ sector, value.var = names(.)[-1][-1])

news_comb_sec <- rbind(news_arc_raw[, .(dte, lbl, sector)], news_arc_raw[, .(dte, lbl, sector)]) %>%
  .[lbl != 0] %>%
  .[, .(fsc_lbl_sum = sum(lbl), fsc_lbl_med = median(lbl)), by = c("dte", "sector")] %>%
  dcast(dte ~ sector, value.var = names(.)[-1][-1])

dat_synth_raw <-
  dat %>%
  merge(., news_eod, by = "dte", all.x = T) %>%
  merge(., news_arc, by = "dte", all.x = T) %>%
  merge(., news_comb, by = "dte", all.x = T) %>%
  merge(., news_eod_sec, by = "dte", all.x = T) %>%
  merge(., news_arc_sec, by = "dte", all.x = T) %>%
  merge(., news_comb_sec, by = "dte", all.x = T) %>%
  setnafill("locf") %>%
  Thesis::create_synthetic_data(., dte_col = "dte")

readr::write_csv(dat_synth_raw[order(dte)], "data/dat_raw.csv")

dat_synth_ft <-
  dat_synth_raw %>%
  add_dataset_features(.)

readr::write_csv(dat_synth_ft, "data/dat_ft.csv")



#
# readr::write_csv(dat_synth_ft[, ..cols], "data/dat_filt.csv")

#### seasonality tests ####

# seastests::combined_test(dat_synth_ft[, tgt_irn], freq = 365)
#
# seastests::combined_test(decompose(ts(dat_synth_ft[, tgt_crd], frequency = 365))$seasonal, freq = 365)

