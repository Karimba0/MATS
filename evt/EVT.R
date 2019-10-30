library(evir)
library(rugarch)
library(ggplot2)
library(xts)
# timesteps prediction in advance
ts <- 5 
# size of fitting window
fit_window = 250
# Date to predict from
date = '2016-01-04'
# distribution for GARCH model, 'std or 'norm
dist <- 'std'
# specify stock here
data <- read.csv('Merge_BKNG.csv')
dates <- data['dt']
index <- which(dates == date)
logreturnsd <- data[c('dt', 'logreturns')][(index-fit_window-ts):nrow(data),]
logreturnsd <- as.data.frame(xts(x=logreturnsd[,-1],order.by= as.POSIXct(logreturnsd$dt)))
names(logreturnsd)[1] <- 'logreturns'
data <- as.data.frame(xts(x=data[,-1],order.by= as.POSIXct(data$dt)))
names(dates)[1] <- 'dt'
index <- which(dates == date)
dates_plot <- as.Date(dates[(index):nrow(data),])

logreturnsm = data.matrix(logreturnsd)
N <- length(logreturnsm)
model=ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = dist
)
#fit and forecast for upper quantiles
fit <- ugarchfit(model, data = logreturnsm, out.sample = N-fit_window)
forc <- ugarchforecast(fit, n.ahead = ts, n.roll = N-fit_window)
z <- (logreturnsm[fit_window:N]-fitted(forc))/sigma(forc)
z <- z['T+5',]

for (alpha in c(0.91, 0.95, 0.975, 0.99)){
  #EVT VaR estimation
  thresh <- findthresh(z, length(z)*0.1)
  out <- gpd(z, thresh)
  tlp <- tailplot(out, optlog = "")
  quantile <- gpd.q(tlp, alpha)
  VAR <- fitted(forc)['T+5',]+(sigma(forc)['T+5',]*quantile[2])
  
  #plot
  VARv <- as.vector(VAR[2:(N-fit_window-ts+1)])
  logreturnsv <- as.vector(logreturnsm[(fit_window+ts+1):N])
  VARd <- data.frame(VARv)
  logreturnsd <- data.frame(logreturnsv)
  logreturns <- logreturnsd
  VaR_t <- VARd
  write.csv(data.frame(logreturns, VaR_t),paste('VaR_',alpha,'.csv',collapse = '', sep = ''), row.names = FALSE)
  p = ggplot() + 
    geom_line(data = logreturnsd, aes(dates_plot, logreturnsv), color = "blue") +
    geom_line(data = VARd, aes(dates_plot, VARv), color = "red") +
    xlab('Date') +
    ylab('logreturns(blue), VaR forecast(red)')
  print(p)
  ggsave(paste('VaR_',alpha,'.png',collapse = '', sep = ''))
  noe <- sum(logreturnsv - VARv < 0)
  print(noe/(length(VARv)))
}

#########################################################

#fit and forecast for lower quantiles
logreturns_negative <- -logreturnsm
fit <- ugarchfit(model, data = logreturns_negative, out.sample = N-fit_window)
forc <- ugarchforecast(fit, n.ahead = ts, n.roll = N-fit_window)
z <- (logreturns_negative[fit_window:N]-fitted(forc))/sigma(forc)
z <- z['T+5',]


for (alpha in c(0.91, 0.95, 0.975, 0.99)){
  #EVT VaR estimation
  thresh <- findthresh(z, length(z)*0.1)
  out <- gpd(z, thresh)
  tlp <- tailplot(out, optlog = "")
  quantile <- gpd.q(tlp, alpha)
  VAR <- fitted(forc)['T+5',]+(sigma(forc)['T+5',]*quantile[2])
  
  #plot
  VARv <- as.vector(VAR[2:(N-fit_window-ts+1)])
  logreturnsv <- as.vector(logreturns_negative[(fit_window+ts+1):N])
  VARd <- data.frame(VARv)
  logreturnsd <- data.frame(logreturnsv)
  logreturns <- -logreturnsd
  VaR_t <- -VARd
  write.csv(data.frame(logreturns, VaR_t),paste('VaR_',1-alpha,'.csv',collapse = '', sep = ''), row.names = FALSE)
  p = ggplot() + 
    geom_line(data = logreturnsd, aes(dates_plot, -logreturnsv), color = "blue") +
    geom_line(data = VARd, aes(dates_plot, -VARv), color = "red") +
    xlab('Date') +
    ylab('logreturns(blue), VaR forecast(red)')
  print(p)
  ggsave(paste('VaR_',1-alpha,'.png',collapse = '', sep = ''))
  noe <- sum(VARv - logreturnsv < 0)
  print(noe/(N-249))
}
