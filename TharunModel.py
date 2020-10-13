import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ridge_regression
import joblib
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import statsmodels.api as sm
import numpy as np

#Data Transform
acca = pd.DataFrame()
for i in range(2011,2016):
    for j in range(1,5):
        acname = 'Acquisition_' + repr(i) + 'Q' + repr(j) + '.txt'
        temp1 = pd.read_csv(acname,sep = '|',header = None)
        tempuse = temp1.iloc[:,[0,4,6,8,18]]
        tempuse.columns = ['ID','Original Balance','Original Date','OriginLTV','State']
        del temp1
        tempuse = tempuse[tempuse['State'] == 'CA']
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
        using2 = pd.merge(left = tempuse, right = period, on = 'Original Date',how = 'inner')
        using2 = using2.reset_index(drop = True)
        acca = pd.concat([acca,using2],axis = 0,ignore_index=True)
# acca.to_csv('accaCA.csv',encoding = 'utf_8_sig')
acca = acca.sample(frac = 0.3)

peca = pd.DataFrame()
for i in range(2011,2016):
    for j in range(1,5):
        acname = 'Performance_' + repr(i) + 'Q' + repr(j) + '.txt'
        temp1 = pd.read_csv(acname,sep = '|',header = None)
        tempuse = temp1.iloc[:,[0,1,4,10]]
        tempuse.columns = ['ID','Current Date','Current Balance','CLDS']
        tempuse = pd.merge(left = tempuse, right = acca, on = 'ID')
        #tempuse = pd.merge(left = tempuse, right = usingtime, on = 'Current Date')
        del temp1
        tempuse = tempuse.reset_index(drop = True)
        peca = pd.concat([peca,tempuse],axis = 0,ignore_index=True)
peca['NLDS'] = peca['CLDS'].shift(-1)
peindex = pd.DataFrame(list(peca.index))
peindex.columns = ['Index']
peca1 = pd.concat([peca,peindex],axis = 1)
droplist = list(peca1.groupby('ID').max()['Index'])
peca1 = peca1.drop(droplist,axis = 0)
indicators = pd.read_csv('indicators.csv')
orihpi = pd.read_csv('orihpi.csv')
peca1 = pd.merge(left = peca1,right = indicators,left_on = 'Current Date',right_on='Date')
peca1['CLTV'] = peca1.apply(lambda x: (x['Current Balance']/x['Original Balance'])*(x['Original HPI']/x['HPI'])*x['OriginLTV'],axis = 1)
period1 = []
for i in range(2005,2016):
    for j in range(1,13):
        if j < 10:
            temp = '0' + repr(j) + '/01/' + repr(i)
            period1.append(temp)
        else:
            temp = repr(j) + '/01/' + repr(i)
            period1.append(temp)
period1 = pd.DataFrame(period1)
period1.columns = ['Current Date']
peca1 = pd.merge(left = peca1,right = period1,on = 'Current Date')

# 'ID', 'Current Date', 'Current Balance', 'CLDS', 'Original Balance',
#        'Original Date', 'NLDS', 'Lumberprice', 'business climate',
#        'new housing', 'ten-year yield', 'unemployment rate', 'CPI', 'GDP',
#        'Current HPI', 'Rental Vacancy Rate', 'Vacant Housing Units for Sale',
#        'CLTV'
pecagroup3 = pd.read_csv('pecaliforniagroup0.csv')
pecagroup3 = pecagroup3.drop(['Unnamed: 0'],axis = 1)
pecagroup3 = pecagroup3.dropna()
peca = pecagroup3.copy()
peca['NLDS'] = peca['NLDS'].map(lambda x: '-1' if x == 'X' else x)
peca = peca[peca['NLDS'] != 'X']
peca['NLDS'] = peca['NLDS'].astype('int32')
peca['NLDS'] = peca['NLDS'].map(lambda x: 4 if x >= 4 else x)
peca3temp1 = peca[peca['NLDS'] == -1]
peca3temp2 = peca[peca['NLDS'] == 0]
peca3temp3 = peca[peca['NLDS'] == 1]
peca3temp1 = peca3temp1.reset_index(drop = True)
peca3temp2 = peca3temp2.reset_index(drop = True)
peca3temp3 = peca3temp3.reset_index(drop = True)
peca3 = pd.concat([peca3temp1,peca3temp2,peca3temp3],axis = 0)
abc = peca3.describe()
peca3['Lumberprice'] = peca3['Lumberprice'].map(lambda x: (x - abc['Lumberprice']['mean'])/abc['Lumberprice']['std'])
peca3['new housing'] = peca3['new housing'].map(lambda x: (x - abc['new housing']['mean'])/abc['new housing']['std'])
peca3['HPI'] = peca3['HPI'].map(lambda x: (x - abc['HPI']['mean'])/abc['HPI']['std'])
peca3['Vacant Housing Units for Sale']  = peca3['Vacant Housing Units for Sale'].map(lambda  x: (x - abc['Vacant Housing Units for Sale']['mean'])/abc['Vacant Housing Units for Sale']['std'])
peca3['business climate'] = peca3['business climate'].map(lambda x: (x - abc['business climate']['mean'])/abc['business climate']['std'])
peca3['unemployment rate'] = peca3['unemployment rate'].map(lambda x: (x - abc['unemployment rate']['mean'])/abc['unemployment rate']['std'])
peca3['Rental Vacancy Rate'] = peca3['Rental Vacancy Rate'].map(lambda x: (x - abc['Rental Vacancy Rate']['mean'])/abc['Rental Vacancy Rate']['std'])
peca3['CLTV'] = peca3['CLTV'].map(lambda x: (x - abc['CLTV']['mean'])/abc['CLTV']['std'])
#VIF
pecagroup = peca3.drop(['ID', 'Current Date', 'Current Balance', 'CLDS', 'Original Balance',
       'Original Date', 'NLDS',],axis = 1)
peca1 = pecagroup.to_numpy()
vif = [variance_inflation_factor(peca1, i) for i in range(peca1.shape[1])]
print(vif)

#Logistic Regression for CLDS == 0
peca = peca3.drop(['ID','Current Date','Current Balance','CLDS','Original Balance','Original Date'],axis = 1)
# peca['NLDS'] = peca['NLDS'] - 3
temp = peca.copy()
temp = temp.dropna()
dy = temp['NLDS']
dx = temp.drop(['NLDS'],axis = 1)
dx = temp.drop(['NLDS','Lumberprice','new housing','HPI','unemployment rate','Rental Vacancy Rate','Vacant Housing Units for Sale'],axis = 1)
dx = dx.astype('float32')
meandx = pd.DataFrame(dx.mean()).T
np.corrcoef(dx)
train_X,test_X,train_y,test_y = train_test_split(dx,dy,test_size=0.1,train_size=0.1,stratify=dy.values)
regressor = LogisticRegression(multi_class='multinomial',solver='saga',penalty='l2',max_iter=300)
regressor.fit(train_X,train_y)
regressor2 = sm.MNLogit(endog=train_y,exog=train_X)
regressor2result = regressor2.fit()
regressor2result.summary()
#regressor.fit(X = train_X,y = train_y)
regressorcv = LogisticRegressionCV(cv = 5,multi_class='multinomial').fit(dx,dy)

coefficent = pd.DataFrame(regressor.coef_)
inter = pd.DataFrame(regressor.intercept_)
model = pd.concat([inter,coefficent],axis = 1)
model.columns = ['Intercept','Lumberprice', 'business climate', 'new housing', 'ten-year yield',
       'unemployment rate', 'CPI', 'GDP', 'HPI', 'Rental Vacancy Rate',
       'Vacant Housing Units for Sale', 'CLTV']
model.to_csv(r'Group0modelcof.csv',encoding = 'utf_8_sig')

coefficent = pd.DataFrame(regressorcv.coef_)
inter = pd.DataFrame(regressorcv.intercept_)
model = pd.concat([inter,coefficent],axis = 1)
model.columns = ['Intercept','Lumberprice', 'business climate', 'new housing', 'ten-year yield',
       'unemployment rate', 'CPI', 'GDP', 'HPI', 'Rental Vacancy Rate',
       'Vacant Housing Units for Sale', 'CLTV']
model.to_csv(r'Group3modelcdcof.csv',encoding = 'utf_8_sig')


ytrainpre = regressor.predict(train_X)
ytestpre = regressor.predict(test_X)
precision_recall_fscore_support(train_y,ytrainpre,average = 'macro')
precision_recall_fscore_support(test_y,ytestpre,average='macro')
regressor.predict_proba(meandx)

ypre = regressorcv.predict(dx)
precision_recall_fscore_support(dy,ypre,average='macro')
regressorcv.predict_proba(meandx)




joblib.dump(regressor,'LRmodel0cv.m')


def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

correlation_matrix(dx)
