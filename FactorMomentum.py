import pandas as pd
import numpy as np

def compute_simple_factor_momentums(data, listOfFactors, lookbackWindow=[12, 1], dateCol='Date'):
    '''compute_simple_factor_momentums takes a data set, and computes the time series and cross sectional factor momentum
    INPUTS:
        data: pandas df, columns should include listOfFactors
        listOfFactors: list, set of factors to include
        lookbackWindow: set of months to use.  Time period to consider it t-lookbackWindow[0], t-lokbackWindow[1]
        dateCol: string, names the date column
    OUTPUTS:
        out: pandas df, should be TSMOM, CSMOM, and Date
            TSMOM: Time series momentum, split into positive, negative and net
            CSMOM: Cross sectional momentum, split into positive, negative and net
            Date: Date Col
    '''
    vals = np.zeros((data.shape[0] - lookbackWindow[0], 6))
    #Fill in Date Column
    for i in range(lookbackWindow[0], data.shape[0]):
        #Compute cumulative return over lookback window
        new = data.loc[i-lookbackWindow[0]:i-lookbackWindow[1], listOfFactors].copy()
        ret = new + 1
        ret = ret.product()
        ret = ret - 1
        #Compute Time Series Momentum
        #Get list of Factors with Positive Return over lookback period
        pos = list(ret[ret > 0].index)
        #Get list of Factors with Negative Return over lookback period
        neg = list(ret[ret < 0].index)
        #Now, compute return on the positive leg of the time series momentum
        if(len(pos) != 0):
            posRet = data.loc[i, pos].mean()
        else:
            posRet = 0
        if(len(neg) != 0):
            negRet = data.loc[i, neg].mean()
        else:
            negRet = 0
        vals[i-lookbackWindow[0],0] = posRet
        vals[i-lookbackWindow[0],1] = negRet
        vals[i-lookbackWindow[0],2] = posRet - negRet
        #Do the same thing with cross sectional factor momentum
        crossPos = list(ret[ret > ret.median()].index)
        crossNeg = list(ret[ret < ret.median()].index)
        crossPosRet = data.loc[i, crossPos].mean()
        crossNegRet = data.loc[i, crossNeg].mean()
        vals[i-lookbackWindow[0],3] = crossPosRet
        vals[i-lookbackWindow[0],4] = crossNegRet
        vals[i-lookbackWindow[0],5] = crossPosRet - crossNegRet
        
    out = pd.DataFrame(vals, columns=['TSMOMPos', 'TSMOMNeg','TSMOMNet',
                                     'CSMOMPos', 'CSMOMNeg', 'CSMOMNet'])
    out['Date'] = data.loc[lookbackWindow[0]:data.shape[0],dateCol].values
    out = out[['Date', 'TSMOMPos', 'TSMOMNeg','TSMOMNet','CSMOMPos', 'CSMOMNeg', 'CSMOMNet']].copy()
    return out

def compute_factor_momentum(data, listOfFactors, lookbackWindow=[12, 1], dateCol='Date', typeOfMOM='TS', method='equal', volHistory=[12,1]):
    '''compute_factor_momentum takes a data set, and computes the time series and cross sectional factor momentum
    INPUTS:
        data: pandas df, columns should include listOfFactors
        listOfFactors: list, set of factors to include
        lookbackWindow: set of months to use.  Time period to consider it t-lookbackWindow[0], t-lokbackWindow[1].  Second argument must be geq 1
        typeOfMOM: string, acceptable inputs are TS or CS.  For time series factor momentum or cross sectional factor momentum
        method: string, acceptable inputs are 'equal' or 'rps' for equal weight or risk parity (simplified), weighting scheme
        volHistory: string, number of data points to use in volatility calculation
    OUTPUTS:
        out: pandas df, should be TSMOM, CSMOM, and Date
            TSMOM: Time series momentum, split into positive, negative and net
            CSMOM: Cross sectional momentum, split into positive, negative and net
            Date: Date Col
    '''
    #Perform basic input checks
    if(dateCol not in list(data.columns)):
        print(dateCol + ' Columm not in data')
        return 0
    if(typeOfMOM not in ['TS','CS']):
        print('Incorrect value for typeOfMOM')
        return 0
    if(method not in ['equal', 'srp']):
        print('Incorrect value for method')
    vals = np.zeros((data.shape[0] - lookbackWindow[0], 3))
    #Fill in Date Column
    for i in range(lookbackWindow[0], data.shape[0]):
        #Compute cumulative return over lookback window
        new = data.loc[i-lookbackWindow[0]:i-lookbackWindow[1], listOfFactors].copy()
        ret = new + 1
        ret = ret.product()
        ret = ret - 1
        #Check Method to define set of factors to use
        if(typeOfMOM == 'TS'):
            #Get list of Factors with Positive Return over lookback period
            pos = list(ret[ret > 0].index)
            #Get list of Factors with Negative Return over lookback period
            neg = list(ret[ret < 0].index)
        elif(typeOfMOM == 'CS'):
            #Get list of factors with above median return over lookback period
            pos = list(ret[ret > ret.median()].index)
            #Get list of factor with below median return over lookback period
            neg = list(ret[ret < ret.median()].index)

        #Now, compute return on the positive leg of the time series momentum
        if(len(pos) != 0):
            if(method== 'equal'):
                posRet = data.loc[i, pos].mean()
            elif(method == 'srp'):
                posVol = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), pos].std()
                weights = 1/posVol
                weights = weights/weights.sum()
                posRet = data.loc[i, pos].transpose().dot(weights)
        else:
            posRet = 0

        if(len(neg) != 0):
            if(method== 'equal'):
                negRet = data.loc[i, neg].mean()
            elif(method == 'srp'):
                negVol = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), neg].std()
                weights = 1/negVol
                weights = weights/weights.sum()
                negRet = data.loc[i, neg].transpose().dot(weights)
        else:
            negRet = 0
        vals[i-lookbackWindow[0],0] = posRet
        vals[i-lookbackWindow[0],1] = negRet
        vals[i-lookbackWindow[0],2] = posRet - negRet

    #Determine the column names
    if(typeOfMOM == 'TS'):
        cols = ['TSMOMPos', 'TSMOMNeg','TSMOMNet']
    elif(typeOfMOM == 'CS'):
        cols = ['CSMOMPos', 'CSMOMNeg', 'CSMOMNet']
        
    out = pd.DataFrame(vals, columns=cols)
    out['Date'] = data.loc[lookbackWindow[0]:data.shape[0],dateCol].values
    cols = ['Date'] + cols
    out = out[cols].copy()
    return out


def compute_portfolio_returns(data, listOfFactors, dateCol, weightingScheme='equal', volHistory=[12,1]):
    '''compute_portfolio_returns takes a set of factors, and a weightingScheme and returns a set of returns
    INPUTS:
        data: pandas df, columns should include the date column, and the listOfFactors
        dateCol: string, names the date column
        weightingScheme: string, must be one of ['equal','srp','cap']
        volHistory: two entry list, defines the lookback window for the calculating volatility
    OUTPUTS:
        out: pandas df, has the date column dateCol, and a second column corresponding to the series of returns specified
    '''
    #Perform Sanity Checks on the Inputs
    if (dateCol not in data.columns):
        print('Date column not in data frame')
        return 0
    if(weightingScheme not in ['equal','srp','cap']):
        print('Incorrect weighting scheme, must be either equal, srp, or cap')
        return 0
    if(volHistory[1] < 1):
        print('bad lookback window, second argumnet must be at least 1')

    #Begin function
    data = data.sort_values(dateCol)
    if(weightingScheme=='equal'):
        out = data.copy()
        out['Equal Weight Returns'] = out[listOfFactors].mean(axis=1)
        l = [dateCol] + ['Equal Weight Returns']
        return out[l]

    elif(weightingScheme=='srp'):
        vals = np.zeros((data.shape[0] - volHistory[0], 1))
        for i in range(volHistory[0], data.shape[0]):
            Vol = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), listOfFactors].std()
            weights = 1/Vol
            weights = weights/weights.sum()
            ret = data.loc[i, listOfFactors].transpose().dot(weights)
            vals[i-volHistory[0]] = ret
        out = pd.DataFrame(vals, columns=['Simple Risk Parity Return'])
        out[dateCol] = data.loc[volHistory[0]:data.shape[0],dateCol].values
        return out[[dateCol,'Simple Risk Parity Return']]

    else:
        #This means you are cap weighted
        weights = pd.DataFrame(1/len(listOfFactors), index=listOfFactors, columns=['Weights'])
        vals = np.zeros((data.shape[0], 1))
        for i in range(data.shape[0]):
            #Compute Returns
            vals[i] = data.loc[i, listOfFactors].dot(weights)
            #Drift Weights
            R = pd.DataFrame(data.loc[i, listOfFactors] + 1)
            R.columns = ['Weights']
            weights = weights*R
            weights = weights/weights.sum()
        out = pd.DataFrame(vals, columns=['Cap Weighted Return'])
        out[dateCol] = data[dateCol]
        return out[[dateCol,'Cap Weighted Return']]

def dynamic_leverage(data, baseCol, colsToLever, dateCol, lookbackWindow=[12,1]):
    '''dynamic leverage scales the returns of colsToLever to match the volatility of baseCol
    INPUTS:
        data: pandas df, needs to contain baseCol, colsToLever and dateCol amoung it's columns
        baseCol: string, names the column to be used as scaling
        colsToLever: columns to lever
        dateCol: date column
        lookbackWindow: array with 2 values, defines the lookback window to adjust the std deviation
    OUTPUTS:
        out: pandas df, contains the dateCol, baseCol, the original unlevered returns, and the leveraged returns, and the leverage ratios
    '''
    vals = np.zeros((data.shape[0] - lookbackWindow[0], 2*len(colsToLever)+1))
    basePlusCols = [baseCol] + colsToLever
    for i in range(lookbackWindow[0], data.shape[0]):
        #Calculate Leverage Using the Lookback Window
        hist = data.loc[i-lookbackWindow[0]:i-lookbackWindow[1], basePlusCols].copy()
        vols = hist.std()
        leverage = vols[baseCol] / vols
        #Store Leverage in the vals array
        vals[i-lookbackWindow[0],0] = vols[baseCol]
        vals[i-lookbackWindow[0],1:len(colsToLever)+1] = leverage[colsToLever]
        vals[i-lookbackWindow[0],len(colsToLever)+1:] = leverage[colsToLever]*data.loc[i,colsToLever]
    
    leverageNames = [name + ' Leverege Ratio' for name in colsToLever]
    leveredNames = [name + ' DL Return' for name in colsToLever]
    cols = [baseCol + ' Lookback Vol'] + leverageNames + leveredNames
    out = pd.DataFrame(vals, columns=cols)
    out['Date'] = data.loc[lookbackWindow[0]:data.shape[0],dateCol].values
    cols = ['Date'] + cols
    out = out[cols].copy()
    return out













