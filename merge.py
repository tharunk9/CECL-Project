import pandas as pd
import datetime as dt

for i in range(2005,2011):
    originalbalance = 0
    currentbalance = pd.DataFrame()
    for j in range(1,5):
        fileac = 'Acquisition_' + repr(i) + 'Q' + repr(j) + '.txt'
        filepe = 'Performance_' + repr(i) + 'Q' + repr(j) + '.txt'
        temp1 = pd.read_csv(fileac,sep = '|',header = None)
        temp2 = pd.read_csv(filepe,sep = '|',header = None)
        using1 = temp2.iloc[:,[0,1,4,10]]
        using1.columns = ['ID','Pay Date','Current Balance','Decline']
        using2 = temp1.iloc[:,[0,4,6]]
        using2.columns = ['ID','Original Balance','Original Date']
        period = []
        for k in range(1,13):
            if k < 10:
                temp = '0' + repr(k) + '/' + repr(i)
                period.append(temp)
            else:
                temp = repr(k) + '/' + repr(i)
                period.append(temp)
        period = pd.DataFrame(period)
        period.columns = ['Original Date']
        using2 = pd.merge(left = using2, right = period, on = 'Original Date')
        originalba = using2['Original Balance']
        originalbanum = sum(originalba)
        originalbalance += originalbanum
        using = pd.merge(left=using2,right = using1,on = 'ID')
        dataset = using
        del using, using1, using2
        dataset['Pay Date'] = pd.to_datetime(dataset['Pay Date'])
        temp3 = dataset.groupby('ID').max()
        temp3 = temp3['Pay Date']
        uniid = pd.DataFrame(temp3.index)
        temp3 = temp3.reset_index(drop = True)
        temp3 = pd.concat([uniid,temp3],axis = 1)
        temp3.columns = ['ID','Pay Date']
        dataset = dataset.drop('Original Date',axis = 1)
        dataset = dataset.reset_index(drop = True)
        dataset1 = pd.merge(left = temp3,right = dataset, on = ['ID','Pay Date'])
        dataset1 = dataset1[dataset1['Decline'] == '3']
        dataset1 = dataset1.drop_duplicates('ID',keep = 'last')
        dataset1 = dataset1.dropna()
        dataset1 = dataset1.reset_index(drop = True)
        lag = dt.timedelta(days = 30)
        for m in range(0,len(dataset1)):
            dataset1['Decline'][m] = int(dataset1['Decline'][m])
        dataset1['Decline'] = dataset1['Decline'] * 30
        datedefault = pd.to_datetime(dataset1['Pay Date']) - pd.to_timedelta(dataset1['Decline'],unit = 'D') - lag
        for l in range(0,datedefault.shape[0]):
            datedefault[l] = datedefault[l].year
        counts = datedefault.value_counts()
        year = pd.DataFrame(counts.index)
        datedefault = pd.DataFrame(datedefault)
        datedefault.columns = ['Year']
        dataset1 = pd.concat([dataset1,datedefault],axis = 1)
        curr = dataset1.groupby('Year').sum()['Current Balance']
        currentbalance = pd.concat([currentbalance,curr],axis = 1)
    currentbalance = currentbalance / originalbalance
    filename = repr(i) + 'Lossbalance3.csv'
    currentbalance.to_csv(filename,encoding = 'utf_8_sig')








