#This library calculates Financial Metrics

import numpy as np #for numerical array data
import pandas as pd #for tabular data

def calc_annualized_mean_return_metrics(data, listOfFactors):
    out = pd.DataFrame((data[listOfFactors].mean() +1)**12-1, columns=['Annual Mean Return'])
    out['Annual SD'] = (12**.5)*data[listOfFactors].std()
    out['t stat'] = data[listOfFactors].mean()/(data[listOfFactors].std()/(data.shape[0]**.5))
    return out

def calc_metrics(data, dictOfPeriods, listOfFactors, interestRate, dateCol, regimeCol, method='sharpe', timeStep='monthly'):
    '''calc_sharpe_ratio_by_regime returns the sharpe ratio, broken out by time period and regime for set of series
    Sharpe Ratio and Log Return are annualized.  Mean return is not
    '''
    if(timeStep not in ['monthly', 'daily', 'yearly']):
        print('Incorrect argument for timeStep')
        return 0

    out = pd.DataFrame(np.zeros((len(listOfFactors),len(list(dictOfPeriods.keys())))),
                        columns=list(dictOfPeriods.keys()), index=listOfFactors)    
    for periodName in dictOfPeriods.keys():
        '''Pull Out the Set of Dates'''
        dateSet = dictOfPeriods[periodName]
        '''Filter The Data'''
        data2 = data.copy()
        data2 = data2[(data2[dateCol] >= dateSet['startDate']) & (data2[dateCol] <= dateSet['endDate'])]
        if(dateSet['Regime'] is not None):
            data2 = data2[data2[regimeCol] == dateSet['Regime']]
        '''Calculate Sharpe Ratio for set of factors'''
        for factor in listOfFactors:
            if(method=='mean'):
                '''Calculate the mean'''
                m = data2[factor].mean()
                out.loc[factor, periodName] = m
            if(method=='sharpe'):
                '''Calculate the sharpe ratio'''
                sharpe = np.mean(data2[factor] - data2[interestRate])/data2[factor].std()
                '''Convert to Annual if desired'''
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = np.sqrt(12)*sharpe
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = np.sqrt(252)*sharpe
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = sharpe

            if(method == 'logReturn'):
                '''Calculate the annualized log return'''
                newData = data2[factor].copy()
                newData = newData+1
                m = np.log(newData).mean()
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = 12*m
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = 252*m
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = m
            if(method=='skew'):
                '''Calculate the skew'''
                skew = data2[factor].skew()
                out.loc[factor, periodName] = skew
            if(method=='kurtosis'):
                '''Calculate the kurtosis'''
                kurt = data2[factor].kurtosis()
                out.loc[factor, periodName] = kurt
    return out

def calc_contagion_measure(data, listOfFactors, listOfRegimeCols):
    out = np.zeros((1,1+2*len(listOfRegimeCols)))
    data2 = data.copy()
    out[0,0] = np.linalg.norm(data2[listOfFactors].corr())/len(listOfFactors)
    count = 1
    l = ['Unconditional']
    for col in listOfRegimeCols:
        conditionalData = data2[data2[col] == 1].copy()
        out[0,count] = np.linalg.norm(conditionalData[listOfFactors].corr())/len(listOfFactors)
        l = l + [col+' = 1']
        count = count + 1
        conditionalData = data2[data2[col] == -1].copy()
        out[0,count] = np.linalg.norm(conditionalData[listOfFactors].corr())/len(listOfFactors)
        count = count + 1
        l = l + [col+' = -1']
    
    out = pd.DataFrame(out, columns=l)
    return out

def calc_metrics_with_nan(data, dictOfPeriods, listOfFactors, interestRate, dateCol, regimeCol, method='sharpe', timeStep='monthly'):
    '''calc_sharpe_ratio_by_regime returns the sharpe ratio, broken out by time period and regime for set of series
    Sharpe Ratio and Log Return are annualized.  Mean return is not
    '''
    if(timeStep not in ['monthly', 'daily', 'yearly']):
        print('Incorrect argument for timeStep')
        return 0

    out = pd.DataFrame(np.zeros((len(listOfFactors),len(list(dictOfPeriods.keys())))),
                        columns=list(dictOfPeriods.keys()), index=listOfFactors)    
    for periodName in dictOfPeriods.keys():
        '''Pull Out the Set of Dates'''
        dateSet = dictOfPeriods[periodName]
        '''Filter The Data'''
        data2 = data.copy()
        data2 = data2[(data2[dateCol] >= dateSet['startDate']) & (data2[dateCol] <= dateSet['endDate'])]
        if(dateSet['Regime'] is not None):
            data2 = data2[data2[regimeCol] == dateSet['Regime']]
        '''Calculate Metric for set of factors'''
        for factor in listOfFactors:
            data3 = data2[[factor, interestRate]].copy()
            data3.dropna(inplace=True)
            if(method=='mean'):
                '''Calculate the mean'''
                m = data3[factor].mean()
                out.loc[factor, periodName] = m
            if(method=='sharpe'):
                '''Calculate the sharpe ratio'''
                sharpe = np.mean(data3[factor] - data3[interestRate])/data3[factor].std()
                '''Convert to Annual if desired'''
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = np.sqrt(12)*sharpe
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = np.sqrt(252)*sharpe
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = sharpe
            if(method == 'logReturn'):
                '''Calculate the annualized log return'''
                newData = data3[factor].copy()
                newData = newData+1
                m = np.log(newData).mean()
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = 12*m
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = 252*m
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = m
            if(method=='skew'):
                '''Calculate the skew'''
                skew = data3[factor].skew()
                out.loc[factor, periodName] = skew
            if(method=='kurtosis'):
                '''Calculate the kurtosis'''
                kurt = data3[factor].kurtosis()
                out.loc[factor, periodName] = kurt
    return out

def compute_individual_factor_ts_momentum(data, listOfFactors, lookbackWindow=[12, 1], dateCol='Date'):
    '''compute_factor_ts_momentum takes a data set, and computes the leg of each factor for TS momentum
    INPUTS:
        data: pandas df, columns should include listOfFactors
        listOfFactors: list, set of factors to include
        lookbackWindow: set of months to use.  Time period to consider it t-lookbackWindow[0], t-lokbackWindow[1]
    OUTPUTS:
        out: pandas df, columns should be the leg (i.e 1 or -1) for that factor in the TS momentum portfolio
    '''
    vals = np.zeros((data.shape[0] - lookbackWindow[0], len(listOfFactors)))
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
        #Now, Mark 1 if in positive leg, otherwise negative
        for j in range(len(listOfFactors)):
            if listOfFactors[j] in pos:
                vals[i-lookbackWindow[0],j] = 1
            else:
                vals[i-lookbackWindow[0],j] = -1
        
    #Now, create the output dataframe
    cols = list()
    for col in listOfFactors:
        cols.append('TSSign'+col)
    out = pd.DataFrame(vals, columns=cols)
    out['Date'] = data.loc[lookbackWindow[0]:data.shape[0],dateCol].values
    cols = ['Date'] + cols
    out = out[cols].copy()
    return out

def calc_conditional_metrics(data, listOfFactors, splitCol, interestRate, namingDict=None, dateCol='Date', method='mean'):
    '''calc_conditional_metrics takes a data set, and computes the conditional financial metrics given the split column
    INPUTS:
        data: pandas df, columns should include listOfFactors and TSSign(theFactor)
        listOfFactors: list, set of factors to include
        interestCol: string, interest rate column
        splitCol: column to be used for conditioning
        namingDict: optional dictionary obejct that changes the names of the columns.  
            Keys are the values in splitCol, values are the names you would like to put into the columns
            
        dateCol: optional string, names the date column
        method: optional string, names the metric to calculate
    OUTPUTS:
        out: pandas df'''
    #Step 1: Store the unique values in the split column
    splitVals = list(data[splitCol].unique())
    splitVals = ['Unconditional'] + splitVals
    vals = np.zeros((len(listOfFactors),len(splitVals)))
    #Step 2: Loop over all of the factors
    for i in range(len(listOfFactors)):
        factor = listOfFactors[i]
        #Loop over the different splits, calculate the metric for this factor
        for j in range(len(splitVals)):
            if(splitVals[j] == 'Unconditional'):
                specificData = data[[factor, interestRate]].copy()
            else:
                specificData = data[[factor, interestRate]][data[splitCol] == splitVals[j]].copy()
            #Calculate the metric
            if(method=='mean'):
                vals[i,j] = (specificData[factor].mean()+1)**12-1            
            if(method=='sharpe'):
                vals[i,j] = np.sqrt(12)*np.mean(specificData[factor] - specificData[interestRate])/specificData[factor].std()
            if(method == 'logReturn'):
                vals[i,j] = 12*np.log(specificData[factor]).mean()      
            if(method == 'skew'):
                vals[i,j] = specificData[factor].skew()
            if(method == 'kurtosis'):
                vals[i,j] = specificData[factor].kurtosis()
            
    #Step 3: Change the column names if specified
    if(namingDict is None):
        out = pd.DataFrame(vals, columns=splitVals,
                      index=listOfFactors)
    else:
        cols = ['Unconditional']
        for j in range(1,len(splitVals)):
            cols.append(namingDict[splitVals[j]])
        out = pd.DataFrame(vals, columns=cols, index=listOfFactors)
    return out