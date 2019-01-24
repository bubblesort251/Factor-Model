#Library for Factor-Model usage


import numpy as np #for numerical array data
import pandas as pd #for tabular data
from scipy.optimize import minimize
import matplotlib.pyplot as plt #for plotting purposes

import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#Plotting Functions
def plot_returns(data, name, flag='Total Return', date='Date'):
    '''plot_returns returns a plot of the returns
    INPUTS:
        name: string, name of column to be plotted
        data: pd dataframe, where the data is housed
        flag: string, Either Total Return or Monthly Return
        date: string, column name corresponding to the date variable
    Outputs:
        a plot'''
    #Clean Inputs:
    if(date not in data.columns):
        print ('date column not in the pandas df')
        return
    if(name not in data.columns):
        print ('column ' + name + ' not in pandas df')
        return
    #If the inputs are clean, create the plot

    if (flag == 'Total Return'):
        n = data.shape[0]
        totalReturns = np.zeros(n)
        totalReturns[0] = 1.
        for i in range(1,n):
            totalReturns[i] = totalReturns[i-1]*(1+data[name][i])
        plt.semilogy(data[date], totalReturns)
        plt.title(name + ' Total Return Over Time')
        plt.ylabel(name)
        plt.xlabel('Date')
        plt.show()
    elif (flag == 'Relative Return'):
        plt.plot(data[date], data[name])
        plt.title(name + ' Returns Over Time')
        plt.ylabel(name)
        plt.xlabel('Date')
        plt.show()
    else:
        print ('flag variable must be either Total Return or Monthly Return')


#Helper Functions
def create_options():
    '''create standard options dictionary to be used as input to regression functions'''
    options = dict()
    options['timeperiod'] = 'all'
    options['date'] = 'Date'
    options['returnModel'] = False
    options['printLoadings'] = True
    return options

def create_options_cv_lasso():
    options = create_options()
    options['maxLambda'] = .25
    options['nLambdas'] = 100
    options['randomState'] = 7777
    options['nFolds'] = 10
    return options


def print_timeperiod(data, dependentVar, options):
    '''print_timeperiod takes a a dependent varaible and a options dictionary, prints out the time period
    INPUTS:
        data: pandas df, df with the data
        dependentVar: string, name of dependent variable
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
    OUTPUTS:
        printed stuff
    '''
    print ('Dependent Variable is ' + dependentVar)
    if(options['timeperiod'] == 'all'):
        sortedValues = data.sort_values(options['date'])[options['date']].reset_index(drop=True)
        n = sortedValues.shape[0]
        beginDate = sortedValues[0]
        endDate = sortedValues[n-1]
        print ('Time period is between ' + num_to_month(beginDate.month) +  ' ' + str(beginDate.year) + ' to ' + num_to_month(endDate.month) +  ' ' + str(endDate.year) + ' inclusive   ')        
    else:
        print ('Time period is ' + options['timeperiod'])

def display_factor_loadings(intercept, coefs, factorNames, options):
    '''display_factor_loadings takes an intercept, coefs, factorNames and options dict, and prints the factor loadings in a readable way
    INPUTS:
        intercept: float, intercept value
        coefs: np array, coeficients from pandas df
        factorNames: list, names of the factors
        options: dict, should contain at least one key, nameOfReg
            nameOfReg: string, name for the regression
    Outputs:
        output is printed
    '''
    loadings = np.insert(coefs, 0, intercept)
    if('nameOfReg' not in options.keys()):
        name = 'No Name'
    else:
        name = options['nameOfReg']
    out = pd.DataFrame(loadings, columns=[name])
    out = out.transpose()
    fullNames = ['Intercept'] + factorNames
    out.columns = fullNames
    print(out)

def best_subset(x,y,l_0):
    # Mixed Integer Programming in feature selection
    M = 1000
    n_factor = x.shape[1]
    z = cp.Variable(n_factor, boolean=True)
    beta = cp.Variable(n_factor)
    alpha = cp.Variable(1)

    def MIP_obj(x,y,b,a):
        return cp.norm(y-cp.matmul(x,b)-a,2)

    best_subset_prob = cp.Problem(cp.Minimize(MIP_obj(x, y, beta, alpha)), 
                             [cp.sum(z)<=l_0, beta+M*z>=0, M*z>=beta])
    best_subset_prob.solve()
    return alpha.value, beta.value


#First function, linear factor model build
def linear_regression(data, dependentVar, factorNames, options):
    '''linear_regression takes in a dataset and returns the factor loadings using least squares regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #first filter down to the time period
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #perform linear regression
    linReg = LinearRegression(fit_intercept=True)
    linReg.fit(newData[factorNames], newData[dependentVar])
    
    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        # Now print the factor loadings
        display_factor_loadings(linReg.intercept_, linReg.coef_, factorNames, options)

    if(options['returnModel']):
        return linReg


def lasso_regression(data, dependentVar, factorNames, options):
    '''linear_regression takes in a dataset and returns the factor loadings using least squares regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            alpha: float, alpha value for LASSO regression
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    if('lambda' not in options.keys()):
        print ('lambda not specified in options')
        return

    #first filter down to the time period
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #perform linear regression
    lassoReg = Lasso(alpha=options['lambda']/(2*data.shape[0]), fit_intercept=True)
    lassoReg.fit(newData[factorNames], newData[dependentVar])
    
    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('lambda = ' + str(options['lambda']))

        #Now print the factor loadings
        display_factor_loadings(lassoReg.intercept_, lassoReg.coef_, factorNames, options)

    if(options['returnModel']):
        return lassoReg

def best_subset_regression(data, dependentVar, factorNames, options):
    '''best_subset_regression takes in a dataset and returns the factor loadings using best subset regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model
            maxVars: int, maximum number of factors that can have a non zero loading in the resulting regression
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #Check dictionary for maxVars option
    if('maxVars' not in options.keys()):
        print ('maxVars not specified in options')
        return

    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #perform linear regression
    alpha, beta = best_subset(data[factorNames].values, data[dependentVar].values, options['maxVars'])
    
    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)

        #Now print the factor loadings
        display_factor_loadings(alpha, beta, factorNames, options)

    if(options['returnModel']):
        return alpha, beta

def cross_validated_lasso_regression(data, dependentVar, factorNames, options):
    '''best_subset_regression takes in a dataset and returns the factor loadings using best subset regression
    INPUTS:
        data: pandas df, data matrix, should constain the date column and all of the factorNames columns
        dependentVar: string, name of dependent variable
        factorNames: list, elements should be strings, names of the independent variables
        options: dictionary, should constain at least two elements, timeperiod, and date
            timeperiod: string, if == all, means use entire dataframe, otherwise filter the df on this value
            date: name of datecol
            returnModel: boolean, if true, returns model

            maxLambda: int, max lambda value passed
            nLambdas: int, number of lambda values to try
            randomState: integer, sets random state seed
            nFolds: number of folds
    Outputs:
        reg: regression object from sikitlearn
        also prints what was desired
    '''
    #Test timeperiod
    if(options['timeperiod'] == 'all'):
        newData = data.copy()
    else:
        newData = data.copy()
        newData = newData.query(options['timeperiod'])

    #Do CV Lasso
    alphaMax = options['maxLambda'] / (2*data.shape[0])
    alphas = np.linspace(1e-6, alphaMax, options['nLambdas'])
    if(options['randomState'] == 'none'):
        lassoTest = Lasso(fit_intercept=True)
    else:
        lassoTest = Lasso(random_state = options['randomState'], fit_intercept=True)

    tuned_parameters = [{'alpha': alphas}]

    clf = GridSearchCV(lassoTest, tuned_parameters, cv=options['nFolds'], refit=False) 
    clf.fit(newData[factorNames],newData[dependentVar])

    alphaBest = alphas[np.argmax(clf.cv_results_['mean_test_score'])]
    lassoBest = Lasso(alpha=alphaBest, fit_intercept=True)
    lassoBest.fit(newData[factorNames],newData[dependentVar])


    if (options['printLoadings'] == True):
        #Now print the results
        print_timeperiod(newData, dependentVar, options)
        print('best lambda = ' + str(alphaBest*2*data.shape[0]))

        #Now print the factor loadings
        display_factor_loadings(lassoBest.intercept_, lassoBest.coef_, factorNames, options)

    if(options['returnModel']):
        return lassoBest

def run_factor_model(data, dependentVar, factorNames, method, options):
    '''run_Factor_model allows you to specify the method to create a model, returns a model object according to the method you chose
    INPUTS:
        data: pandas df, must contain the columns specified in factorNames and dependentVar
        dependentVar: string, dependent variable
        factorNames: list of strings, names of independent variables
        method: string, name of method to be used.  Supports OLS, LASSO, CVLASSO atm
        options: dictionary object, controls the hyperparameters of the method
    Outputs:
        out: model object'''

    #Make sure the options dictionary has the correct settings
    options['returnModel'] = True
    options['printLoadings'] = False

    #Now create the appropriate model
    if (method == 'OLS'): #run linear model
        return linear_regression(data, dependentVar, factorNames, options)
    if (method == 'LASSO'):
        return lasso_regression(data, dependentVar, factorNames, options)
    if (method == 'CVLasso'):
        return cross_validated_lasso_regression(data, dependentVar, factorNames, options)
    else:
        print ('Method ' + method + ' not supported')


#Asorted Nonsense
def num_to_month(month):
    #num to month returns the name of the month, input is an integer
    if (month==1):
        return 'January'
    if (month==2):
        return 'Febuary'
    if (month==3):
        return 'March'
    if (month==4):
        return 'April'
    if (month==5):
        return 'May'
    if (month==6):
        return 'June'
    if (month==7):
        return 'July'
    if (month==8):
        return 'August'
    if (month==9):
        return 'September'
    if (month==10):
        return 'October'
    if (month==11):
        return 'November'
    if (month==12):
        return 'December'


def data_time_periods(data, dateName):
    '''data_time_periods figures out if the data is daily, weekly, monthly, etc
    INPUTS:
        data: pandas df, has a date column in it with column name dateName
        dateName: string, name of column to be analysed
    '''
    secondToLast = data[dateName].tail(2)[:-1]
    last = data[dateName].tail(1)
    thingy = (last.values - secondToLast.values).astype('timedelta64[D]') / np.timedelta64(1, 'D')
    thingy = thingy[0]
    if (thingy > 200):
        return 'yearly'
    elif(thingy > 20):
        return 'monthly'
    elif(thingy > 5):
        return 'weekly'
    else:
        return 'daily'




















