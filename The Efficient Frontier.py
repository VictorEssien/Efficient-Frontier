#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as scs
import scipy.optimize as sco


# In[2]:


def ticker():
    """ 
    Gets the number of assets, and assets to add in a portfolio from the user
        Returns the tickers of the assets
    """
    ticker = []
    num_tickers = int(input("How many assets will be in your portfolio? "))
    
    for num in range(num_tickers):
        tick = input("What asset would you like to add in your portfolio? ").upper()
        ticker.append(tick)
    return ticker


# In[3]:


def get_data(tickers):
    """ 
    Gets the time from the user, downloads the Adjusted Close of tickers 
        and returns the data 
    """
    
    time = int(input("How many years worth of data do you want? "))
    end = dt.datetime.today()
    start = end - dt.timedelta(365 * time)
    
    ticker_data = yf.download(tickers, start, end)["Adj Close"]
    data_output = pd.DataFrame(ticker_data)
    
    return data_output


# In[4]:


tickers = ticker()


# In[5]:


tickers


# In[6]:


data = get_data(tickers)
data


# In[7]:


returns = data.pct_change()
returns


# In[8]:


#Mean Returns
mean = returns.mean() * 100
mean


# In[9]:


#The Covariance Matrix
cov = returns.cov() * 100
cov


# In[10]:


#Plotting the Covariance Matrix

plt.style.use("ggplot")

fig = plt.figure(figsize=(10,len(tickers)))

sns.heatmap(cov,
           cmap="OrRd",
           annot=True,
           linewidth=0.4)

plt.title("Covariance of the Returns\n")
plt.savefig("The Covariance Returns.pdf")
plt.show()


# In[13]:


class Efficient_Frontier:
    
    def __init__(self, tickers):
        
        #Input
        self.tickers = tickers
    
    def portfolio_returns(self, weights):
        """
        Returns the annualized (expected) portfolio return,
            given the portfolio weights
        """
        self.weights = weights
        
        return np.sum(returns.mean() * weights) * 252
    
    def portfolio_volatility(self, weights):
        """ 
        Returns the annualized portfolio volatility, 
            given the portfolio weights
        """
        self.weights = weights
        
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    def plot_simulation(self):
        """ 
        Uses Monte Carlo simulation to plot the expected return 
            and volatility for random portfolio weights
        """
        portfolio_returns = self.portfolio_returns
        portfolio_volatility = self.portfolio_volatility
        
        port_returns = []
        port_volatility = []
        size = len(tickers)
        
        for i in range(50000):
            weights = np.random.random(size)  #Randomizing the portfolio weights
            weights /= np.sum(weights)      #Normalized to 1 or 100%
            
            port_returns.append(portfolio_returns(weights))
            port_volatility.append(portfolio_volatility(weights))
            
        port_returns = np.array(port_returns)
        port_volatility = np.array(port_volatility)
        
        plt.figure(figsize=(12, len(tickers)))
        plt.scatter(port_volatility, port_returns, c=port_returns/port_volatility,
                    marker = "o", cmap="coolwarm")
        plt.xlabel("Expected Volatility")
        plt.ylabel("Expected Return")
        plt.title("The Monte Carlo Simulation")
        plt.colorbar(label="Sharpe Ratio");
        
        
    def minimize_sharpe(self, weights):
        """ Function to maximize the Sharpe Ratio"""
        self.weights = weights
        portfolio_returns = self.portfolio_returns
        portfolio_volatility = self.portfolio_volatility
        
        return -portfolio_returns(weights) / portfolio_volatility(weights)
        
        
    def plot_frontier(self):
        """
        Plots the Efficient Frontier for the optimal portfolios,
            given a certain target return
        """
        
        portfolio_returns = self.portfolio_returns
        
        portfolio_volatility = self.portfolio_volatility
        
        minimize_sharpe = self.minimize_sharpe
            
        size = len(tickers)
        
        
        port_returns = []
        port_volatility = []
        
        for i in range(50000):
            weights = np.random.random(size)   #Randomizing the portfolio weights
            weights /= np.sum(weights)         #Normalized to 1 or 100%
            
            port_returns.append(portfolio_returns(weights))
            port_volatility.append(portfolio_volatility(weights))
            
        port_returns = np.array(port_returns)
        port_volatility = np.array(port_volatility)
            
        #Getting the maximum Sharpe value (Maximizing the Sharpe ratio) and the optimal portfolio composition
        #by minimizing the negative value of the Sharpe ratio

        #The constraint is that all parameters (weights) add up to 1.
        
        
        eweights = np.array(size * [1. / size,])  #Equal Weight Vectors
            
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})  #Equality constraints
            
        bnds = tuple((0, 1) for x in range(size))   #Bounds for parameters
            
        opts = sco.minimize(minimize_sharpe, eweights,
                            method="SLSQP", bounds=bnds,
                            constraints=cons)
        
        maximum_sharpe = portfolio_returns(opts["x"]) / portfolio_volatility(opts["x"]) 
        optimal_weights = [opts["x"].round(3)]
        
        
        #The minimization of the variance of the portfolio (i.e. minimizing the volatility)
        opt_vol = sco.minimize(portfolio_volatility, eweights,
                               method='SLSQP', bounds=bnds,
                               constraints=cons)
            
            
        #Equality constraints and the bounds for the parameters
        constraint = ({'type': 'eq', 'fun': lambda x: portfolio_returns(x) - tret},
                      {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
        bound = tuple((0, 1) for x in weights)
        
        
        target_return = np.linspace(0.25, 2.03, 45)
        target_volatility = []
        
        
        for tret in target_return:
            res = sco.minimize(portfolio_volatility, eweights, method='SLSQP',
                               bounds=bound, constraints=constraint)
                
            target_volatility.append(res['fun'])
        
        target_volatility = np.array(target_volatility)
                
        print("The maximum Sharpe ratio is {}".format(maximum_sharpe))
        print("The optimal portfolio weights are:{}".format(list(optimal_weights)))
        
        plt.figure(figsize=(12, len(tickers)))
        plt.scatter(port_volatility, port_returns, c=port_returns/port_volatility,
                    marker='.', alpha=0.9, cmap='coolwarm')
            
        plt.plot(target_volatility, target_return, 'b', lw=3.5)
        plt.plot(portfolio_volatility(opts['x']), portfolio_returns(opts['x']),   #Minimum Variance (Return)
                 'r*', markersize=18.0)
        plt.plot(portfolio_volatility(opt_vol['x']), portfolio_returns(opt_vol['x']),  #Maximum Sharpe (return)
                 'g*', markersize=18.0)
        plt.xlabel('Expected volatility')
        plt.ylabel('Expected return')
        plt.title("The Efficient Frontier")
        plt.colorbar(label='Sharpe ratio')
        plt.savefig("The Efficient Frontier.pdf");


# In[12]:


sim = Efficient_Frontier(tickers)
sim.plot_simulation()


# In[14]:


frontier = Efficient_Frontier(tickers)
frontier.plot_frontier()


# In[ ]:




