# Begin analysis
# Load required packages 
suppressPackageStartupMessages(library("argparse"))
library(ggplot2) 
library(R.utils) 
library(accelerometry) 
library(plyr) 
library(dplyr)
library(lubridate) 
library(raster) 
library(maps) 
library(gridExtra) 
library(viridis) 
library(RColorBrewer) 
library(zoo) 
library(nlme)
library(tibble)

# # create parser object
# parser <- ArgumentParser()
# 
# parser$add_argument("-t", "--trial", type="double")
# args <- parser$parse_args()

# change working directory. Don't put the trial num in the string because we are pasting it
trial_num = 10
setwd(paste0("~/Desktop/ProjectX/datasets/trial", toString(trial_num)))
filename = "73.125_18.8143.csv"


CReg <- read.csv(Sys.glob("*.csv"))

CReg$date <- as.POSIXct(paste(CReg$ValidDate, CReg$ValidTime), tz="EST")
original_date <- CReg$date
droppings = c("ValidDate", "ValidTime", "Longitude", "Latitude")
CReg = CReg[, !(colnames(CReg) %in% droppings)]
seq1 <- zoo(order.by=as.POSIXct(seq(min(CReg$date), max(CReg$date), by=3600)))
mer1 <- merge(zoo(x=CReg, order.by=CReg$date), seq1)
mer1 = mer1[, !(colnames(CReg) %in% c("date"))]
mer1 <- mer1[6:nrow(mer1)]
mer1 <- mer1[1:nrow(mer1) - 1]

dataL <- na.approx(mer1)
CReg <- fortify.zoo(dataL, name = "date")
CReg[, 2:4] <- CReg[, 2:4] %>% mutate_if(is.character,as.numeric)

p_opt <- c(32.6, 1.76, 16.6, 27.2, 30.3, 37.6)

# Temperature dependent rate function
# (taking Temperature and p = c(tmin, topt, tmax))
rt <- function(Temp, p) {
  r <- ((p[3] - Temp) / (p[3] - p[2])) * ((Temp - p[1]) / (p[2] - p[1])) ^ ((p[2] - p[1]) / (p[3] - p[2]))
  r[r < 0] <- 0 # set negative values to zero
  r[is.na(r)] <- 0 # set undefined values to zero
  return(r) 
}

# Ft with temperature-dependent rate function
# First argument is vector of times
Ft <- function(hr, Temp, p) {
  # p = c(alpha, gamma, tmin, topt, tmax, beta) where beta is scaling factor 
  # Temperature-dependent rate
  r <- rt(Temp, p[3:5])
  # Estimated H(t, T)
  Ht <- r * (hr / p[1]) ^ p[2] 
  # F(t, T)
  return((1 - exp(-Ht))*p[6])
}

  
# # Cumulative Distribution Function with temperature-dependent rate
# Ftr <- function(ct, p_opt){ # takes canopy temperatures during a wet period as input
#   w <- length(ct) - 1 # LWD
#   ti <- matrix(ct[1:w] + diff(ct)/2, nc = w, nr = w, byrow = F) # mean temperatures 
#   ri <- rt(ti, p_opt[3:5]) # relative rates
#   im <- matrix(1:w, nc = w, nr = w) - rep((1:w) - 1, each= w) # end times per cohort 
#   im0 <- im-1 # start times
#   a <- p_opt[1]; g <- p_opt[2] # Weibull survival parameters
#   Hm <- ri*((im/a)^g - (im0/a)^g) # matrix giving H for each cohort and interval 
#   Hm[im0 < 0] <- 0 # replace values with zeros when t < 0 per cohort
#   Hmc <- apply(Hm, 2, cumsum) # cumulative hazard across intervals
#   Fm <- 1-exp(-Hmc) # cumulative probability matrix by cohort
#   return(rowSums(Fm)[w]) # total relative number of infecting cohorts
# }

# Function to create matrix of wet periods from vector of canopy wetness
wet.per <- function(wet){
  x <- wet > 0
  m <- rle2(as.numeric(x), indices = TRUE) # rle2 function for run lengths 
  m <- matrix(m[m[,1] == 1,], nc = 4)
  return(m)
}

# Function to calculate infections from:
# CM = canopy moisture 3 hr intervals, RH = relative humidity 3 hr intervals,
# CT = canopy temperature 3 hr intervals, p_opt = vector of optimized parameters, # Returns total infections for timeseries
create_infect <- function(data, p_opt){ # uses 3-hourly estimates
  if(all(is.na(data$CT)))  return(NA) 
  data$CT <- if(all(data$CT < 100)) data$CT else data$CT - 273 # convert from Kelvin if required
  wet <- data$CM > 0 | data$RH >= 98 # classify each hour as wet or dry
  if(all(wet == FALSE)) return (0) # stop calculations if always dry
  wetper <- wet.per(wet) # identify wet and dry hours
  wetper <- wetper[wetper[,4] > 3,] # keep only wet periods above 3 hours duration 
  wetper <- matrix(wetper, nc = 4) # reformat
  if(nrow(wetper) == 0) return (0) # stop if no long wet periods
  res <- rep(0, nrow(data)) # list for results
  for(i in 1:nrow(wetper)){ # for each wet period...
    cti <- data$CT[wetper[i,2]:wetper[i,3]] # take canopy temperatures for that period
    time = seq(1:length(cti))
    Fti <- Ft(time, cti, p_opt) # calculate infections for that period
    res[wetper[i,2]:wetper[i,3]] = Fti # add result to list
  }
  return(round(res, 2)) # vector of the total infections per wet period 
}

  
# Run with example data
CReg$num_infect = create_infect(CReg, p_opt)
CReg2 <- filter(CReg, date %in% original_date)

write.csv(CReg2, filename, row.names=F)
  







