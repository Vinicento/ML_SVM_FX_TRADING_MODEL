import pandas as pd
import numpy as np
import datetime
from sklearn import svm,tree
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report, precision_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

""" 
Określenie problemu:
prognozowanie kótkoterminowych trendów cenowych przy użyciu klasyfikacji
"""
#odczytanie pliku i dobranie odpowiednich atrybutów
#preprocessing: uzupełnienie braków ( f fill z powodu działania na małym interwale czasowym i najprawdopodobniej brakujące dane są bardzo bliskie lub takie jak sąsiednie) 
def extract_hour_minute(timestamp_str):
    timestamp_obj = datetime.datetime.strptime(timestamp_str, "%H:%M:%S.%f")
    hour = timestamp_obj.hour
    minute = timestamp_obj.minute
    numeric_value = hour * 100 + minute
    return numeric_value
def model_data_preprocess(data_tick):

    GBPUSD_tick = pd.read_csv("GBPUSD.csv", sep="\t")   
    GBPUSD_tick=GBPUSD_tick[['<TIME>','<BID>','<ASK>']] 
    GBPUSD_tick = GBPUSD_tick.fillna(method='ffill')
    GBPUSD_tick=GBPUSD_tick.loc[0:100000]

    # zmiana wartości z konkretnych kwot na procent zmiany
    GBPUSD_tick["ask_change"]=GBPUSD_tick["<ASK>"].pct_change(fill_method=None)
    GBPUSD_tick["bid_change"]=GBPUSD_tick["<BID>"].pct_change(fill_method=None)


    #Atrybuty : średnie
    GBPUSD_tick['price']=GBPUSD_tick['<ASK>']+GBPUSD_tick['<BID>']/2
    # wybrane zostało kilka okresów i policzona została procentowa odległość ceny od średniej  
    GBPUSD_tick['sma500_distance']=1-(GBPUSD_tick["price"].rolling(500).mean()/GBPUSD_tick['price'])
    GBPUSD_tick['sma200_distance']=1-(GBPUSD_tick["price"].rolling(200).mean()/GBPUSD_tick['price'])
    GBPUSD_tick['sma50_distance']=1-(GBPUSD_tick["price"].rolling(50).mean()/GBPUSD_tick['price'])
    GBPUSD_tick['sma20_distance']=1-(GBPUSD_tick["price"].rolling(20).mean()/GBPUSD_tick['price'])

    # oraz średnia dla procentowej zmiany ceny rozdzielona na cenę kupna i sprzedaży
    GBPUSD_tick['sma50_ask']=GBPUSD_tick["ask_change"].rolling(50).mean()
    GBPUSD_tick['sma50_bid']=GBPUSD_tick["bid_change"].rolling(50).mean()
    GBPUSD_tick['sma150_ask']=GBPUSD_tick["ask_change"].rolling(150).mean()
    GBPUSD_tick['sma150_bid']=GBPUSD_tick["bid_change"].rolling(150).mean()

    # pozostałe atrybuty
    stoch_rsi_ask = ta.momentum.StochRSIIndicator(GBPUSD_tick['ask_change'], 140, 30, 30, False)
    stoch_rsi_bid = ta.momentum.StochRSIIndicator(GBPUSD_tick['bid_change'], 140, 30, 30, False)

    aaron_ask=ta.trend.AroonIndicator(GBPUSD_tick['ask_change'],250)
    aaron_bid=ta.trend.AroonIndicator(GBPUSD_tick['bid_change'],250)

    GBPUSD_tick['aaron_bid']=aaron_bid.aroon_indicator()
    GBPUSD_tick['aaron_ask']=aaron_ask.aroon_indicator()

    GBPUSD_tick['roc_ask']=ta.momentum.roc(GBPUSD_tick['ask_change'], 140)

    GBPUSD_tick['ulcer_index_ask']=ta.volatility.ulcer_index(GBPUSD_tick['ask_change'], 140)
    GBPUSD_tick['ulcer_index_bid']=ta.volatility.ulcer_index(GBPUSD_tick['bid_change'], 140)

    GBPUSD_tick['stoch_ask']=stoch_rsi_ask.stochrsi()
    GBPUSD_tick['stoch_bid']=stoch_rsi_bid.stochrsi()

    GBPUSD_tick["std200_ask"]=1-(GBPUSD_tick["ask_change"].rolling(200).std()/GBPUSD_tick['ask_change'])
    GBPUSD_tick["std200_bid"]=1-(GBPUSD_tick["bid_change"].rolling(200).std()/GBPUSD_tick['bid_change'])

    #zamiana atrybutu czasu na wartość liczbową
    GBPUSD_tick["<TIME>"]=GBPUSD_tick["<TIME>"].apply(extract_hour_minute)
    # usunięcie zbędnych kolumn oraz zamiana wartoći nieskońćzonych
    GBPUSD_tick = GBPUSD_tick.drop(columns=['<BID>', '<ASK>'])
    GBPUSD_tick = GBPUSD_tick.replace([np.inf, -np.inf], np.nan)
    GBPUSD_tick = GBPUSD_tick.fillna(method='ffill')
    #usuniecie wierszy pustych ( ostatnie wiersze dla których nie dało się policzyć wskaźników)
    GBPUSD_tick.dropna(inplace=True)

        #Skalowanie 
    X = GBPUSD_tick.drop(columns=['pred', 'price'])
    y = GBPUSD_tick['pred'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

data_scaled = pd.DataFrame(X_scaled, columns=X.columns)


#Sprawdzenie korelacji atrybutów 
corr_matrix = GBPUSD_tick.corr()
fig, ax = plt.subplots(figsize=(40, 10))
im = ax.imshow(corr_matrix.values, cmap='coolwarm')
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90) 
ax.set_yticklabels(corr_matrix.columns)
cbar = ax.figure.colorbar(im, ax=ax)
plt.show()



#podział na zbiór testowy i trenujący
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.4, random_state=0)

#lista algorytmów do porównania
results={}
classsifiers={}
classsifiers["AdaBoostClassifier"]=AdaBoostClassifier()
classsifiers["BaggingClassifier"]=BaggingClassifier()
classsifiers["svc"]=svm.SVC(kernel="rbf")

""" 
trenowanie algorytmów z podzieleniem na wygrane i przegrane 
czyli z trzech możliwych klas: 1,wzrosty, 2.spadki, 3.brak, stratne to przypadki błędnie zaklasyfikowane jako wzrost lub spadek( w tych przypadkach zostaje zawarta pozycja) 
błedna klasyfikacja jako 3.brak jest drugorzędna
zwrócona zostaje:
-ilość poprawnie zaklasyfikowanych jako wzrost lub spadek ( potencjalnie zyskowne transakje)
-ilość błednie zaklasyfikowanych jako wzrost lub spadek ( potencjalnie stratne transakcje) 
-procent zyskownych
 """

k_fold = KFold(10, shuffle=True,)
y=pd.Series(y)
X = X.reset_index()

for classifier_name, classifier in classsifiers.items():

    for train_ix, test_ix in k_fold.split(X):
        X_train, y_train, X_test, y_test = X.loc[train_ix], y.loc[train_ix], X.loc[test_ix], y.loc[test_ix]
        model=classifier.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        win=cm[1][1]+cm[2][2]
        loss=cm[1][0]+cm[2][0]+cm[2][1]+cm[1][2]
        percent_win=win/(win+loss)
        if classifier_name in results:
            results[classifier_name].append([win, loss, percent_win])
        else:
            results[classifier_name] = [[win, loss, percent_win]]


for classifier_name, classifier in classsifiers.items():
    results[classifier_name]=[sum(x)/10 for x in zip(*results[classifier_name])]

#wyniki
names=results.keys()
values=results.values()
values=[x[2] for x in values]
fig, axes = plt.subplots(figsize=(8, 8))
axes.bar(names,values)
plt.legend()
plt.show()



