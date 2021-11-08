import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks
import warnings
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_ind
warnings.filterwarnings("ignore")

######################################
# Begin of Utils
######################################

def gaussian_kernel(n):
    x = np.linspace(-n, n, 2*n + 1)
    sigma = (2 * n + 1) / 6;
    y = 1 / (np.sqrt(np.pi) * sigma) * np.exp(-((x / sigma) ** 2) / 2)
    y = y / np.sum(y)
    return y

def process_data(temp, precip, base, n):
    nrow, ncol = temp.shape #Year, Days of Year
    filt_t = np.zeros((nrow, n, ncol))
    filt_p = np.zeros((nrow, n, ncol))

    old_t = np.copy(temp);
    old_p = np.copy(precip);
    for i in range(0, n-1): #Power
        j = int(np.round(base**(i+1)))
        kg = gaussian_kernel(j)
        displ = max(((2*j+1 - ncol)//2,0))
        for l in range(0, nrow): # Year
            c_t = np.convolve(temp[l,:], kg, mode='same')[displ:(displ+ncol)]
            c_p = np.convolve(precip[l,:], kg, mode='same')[displ:(displ+ncol)]
            

            filt_t[l, i, :] = old_t[l,:] - c_t
            filt_p[l, i, :] = old_t[l,:] - c_p
        
            old_t[l,:] = c_t
            old_p[l,:] = c_p
    for l in range(0, nrow):
        filt_t[l, n-1, :] = old_t[l,:]
        filt_p[l, n-1, :] = old_t[l,:]
        
    return filt_t, filt_p

def pack_data(x_temp, x_precip, start_date, n_days, l):
    year, length, days = x_temp.shape
    h_size = n_days * (l[1] - l[0])
    v_size = 2 * h_size
    
    x_data = np.zeros((year, v_size))
    
    ia = l[0]
    ib = l[1]
    
    ja = min(start_date, days)
    jb = min(start_date + n_days, days)
    
    size = min((ib-ia)*(jb-ja), h_size)
    x_data[:, 0:size] = x_temp[:, ia:ib, ja:jb].reshape(year, -1)
    
    size = min((ib-ia)*(jb-ja) + h_size, v_size)
    x_data[:, h_size:size] = x_precip[:, ia:ib, ja:jb].reshape(year, -1)
    return x_data

"""
Diffusion Maps class scikit-learn style

eps - floating point value, default=None(automatic)
      Epsilon power to which the distance matrix is raised. 
      It controls the ammount of connections between points
      
features - tuple (a,b) int values, default=None(use all features)
           Selects which range of features are returned when tranforming the data. 
           When None is set, return all available features.
           Given data with n samples and m features, ie, X(n,m), the maximun
           number of available features is **n**(yes n 'samples' not m 'features'
           for this transform)
"""
class DiffusionMaps(BaseEstimator, TransformerMixin):
    
    def __init__(self, eps=None, features=None, target_std=0.1, min_eps=0.001, max_eps=1000):
        self._eps = eps
        if(features is not None):
            features = [i for i in range(features[0], features[1])]
            
        self._features = features
        self._target_std = target_std
        self._min_eps = min_eps
        self._max_eps = max_eps
        
    def fit(self, X, y = None ):
        self.X = X.copy()
        D = cdist(X, X, 'sqeuclidean')
        self.D=D
        if(self._eps is  None):
            self._eps = self.__find_eps()
            
        E = np.exp(-D / (self._eps**2))
        self.P = normalize(E, axis=1, norm='l1')     
        e, V = np.linalg.eig(self.P)
        idx = np.argsort(e.real)
        e = e.real[idx[::-1]]
        V = V.real[:, idx[::-1]]
        self.e = e
        self.V = V
        return self
    
    def transform(self, X, y = None):
        D = cdist(X, self.X, 'sqeuclidean')
        E = np.exp(-D / (self._eps**2))
        P = normalize(E, axis=1, norm='l1')
        if(self._features is None):
            return P.dot(self.V).dot(np.diag(1.0 / self.e))
        else:
            return P.dot(self.V).dot(np.diag(1.0 / self.e))[:, self._features]
        
    def __find_eps(self):
        n = self.D.shape[0]
        target = (1 / n) * self._target_std / 6;
        i = 0
        maxit = 100
        a = self._max_eps
        b = self._min_eps

        P = normalize(np.exp(-self.D / (a**2)), axis=1, norm='l1')
        fa = np.std(P.ravel())

        P = normalize(np.exp(-self.D / (b**2)), axis=1, norm='l1')
        fb = np.std(P.ravel())

        while((abs(fa - fb) > (target/50)) and i < maxit):

            c = (a+b)/2 
            i = i + 1
            P = normalize(np.exp(-self.D / (c**2)), axis=1, norm='l1')
            fc = np.std(P.ravel())
            if(fc < target):
                a = c
                fa = fc
            else:
                b = c
                fb = fc
        if(i >= maxit):
            print("(Diffusion Maps)Reached max iterations without finding eps")
                
        return c
    
    def getPOrdered(self):
        idx = np.argsort(self.V[:,1])
        P = self.P[:, idx]
        P = P[idx, :]
        return P
    
    def getP(self):
        return self.P
    
    def hist(self):
        y = np.sort(self.D.reshape(-1))
        x = np.linspace(0,1, num=len(y))
        plt.plot(x,y)
        
    def getE(self):
        return self.e

def make_stats(a1, b1, gd1, n, tag):
    al_train = np.array(0)
    bl_train = np.array(0)

    al_out = np.array(0)
    bl_out = np.array(0)

    for key, value in gd1.items():
        a_out = a1[key][n:]
        b_out = b1[key][n:]
        gd_out = gd1[key][n:]

        a_train = a1[key][:n]
        b_train = b1[key][:n]
        gd_train = gd1[key][:n]

        at_out = np.zeros(len(a_out))
        bt_out = np.zeros(len(b_out))

        at_train = np.zeros(len(a_train))
        bt_train = np.zeros(len(b_train))

        at_out[a_out == gd_out] = 1
        bt_out[b_out == gd_out] = 1

        at_train[a_train == gd_train] = 1
        bt_train[b_train == gd_train] = 1

        al_out = np.append(al_out, at_out)
        bl_out = np.append(bl_out, bt_out)

        al_train = np.append(al_train, at_train)
        bl_train = np.append(bl_train, bt_train)
        
    c_out = mcnemar(confusion_matrix(al_out, bl_out, labels=(0,1)))
    result_mc_out = {'tag': tag, 'set': 'out-of-sample', 'sample-count': len(al_out),'type': 'McNemar', 'pvalue': c_out.pvalue, 'statistic': c_out.statistic }

    c_train = mcnemar(confusion_matrix(al_train, bl_train, labels=(0,1)))
    result_mc_train = {'tag': tag, 'set': 'train', 'sample-count': len(al_train), 'type': 'McNemar', 'pvalue': c_train.pvalue, 'statistic': c_train.statistic }

    stats_out, pvalue_out = ttest_ind(al_out, bl_out, alternative='greater')
    result_st_out = {'tag': tag, 'set': 'out-of-sample', 'sample-count': len(al_out), 'type': 'Ttest', 'pvalue': pvalue_out, 'statistic': stats_out }

    stats_train, pvalue_train = ttest_ind(al_train, bl_train, alternative='greater')
    result_st_train = {'tag': tag, 'set': 'train', 'sample-count': len(al_train), 'type': 'Ttest', 'pvalue': pvalue_train, 'statistic': stats_train }
    
    return result_st_train, result_st_out, result_mc_train, result_mc_out



######################################
# End of Utils
######################################

######################################
# Begin of Stats Generation
######################################

#input files
path = os.path.join("..", "data");
temp_file = "temp_avg.csv"
precip_file = "precip.csv"
dengue_file = "dengue.csv"
results_file = os.path.join("..", "results", "result.csv")
stats_file = os.path.join("..", "results", "stats.csv")

results = pd.read_csv(results_file)

ground_truth = {}
our_prediction = {}
rnd_prediction = {}
moda_prediction = {}

##
#Run for each city
##
for i, result in results.iterrows():
    ##
    #Read csv files
    ##
    which_state = result["city"]
    temp = pd.read_csv(os.path.join(path, which_state,temp_file))
    precip = pd.read_csv(os.path.join(path, which_state,precip_file))
    dengue = pd.read_csv(os.path.join(path, which_state,dengue_file))
    ##
    #get out of sample test data
    ##
    ##
    #get out of sample test data
    ##
    precip_out = precip.iloc[11:]
    temp_out = temp.iloc[11:]
    dengue_out = dengue.iloc[11:]
    ##
    #get train data
    ##
    precip = precip.iloc[0:11]
    temp = temp.iloc[0:11]
    dengue = dengue.iloc[0:11]

    t_m = np.mean(temp.values)
    t_s = np.std(temp.values)

    p_m = np.mean(precip.values)
    p_s = np.std(precip.values)

    ##
    #Process the data
    ##
    outbreak_threshold = 100.0 #incidence threshold
    x_temp = (temp.values - t_m) / t_s
    x_precip = (precip.values - p_m) / p_s

    y_dengue = np.zeros((len(dengue['incidence'])), dtype=int)
    y_dengue[dengue['incidence'] >= outbreak_threshold] = 1

    x_temp_out = (temp_out.values - t_m) / t_s
    x_precip_out = (precip_out.values - p_m) / p_s

    y_out = np.zeros((len(dengue_out['incidence'])), dtype=int)
    y_out[dengue_out['incidence'] >= outbreak_threshold] = 1

    #Data processing parameters
    base = np.sqrt(2) #filter suport base
    n_levels = 18 #number of levels to generate
    x_t_train, x_p_train = process_data(x_temp, x_precip, base, n_levels)
    x_t_out, x_p_out = process_data(x_temp_out, x_precip_out, base, n_levels)

    ##
    # Figure Generation
    ##

    n_days = result["ndays"]
    l = (result["band_a"], result["band_b"])
    features = (result["feat_a"], result["feat_b"])
    start_day = result["index"]

    X_train = pack_data(x_t_train, x_p_train, start_day, n_days, l)
    Y_train = y_dengue

    X_out = pack_data(x_t_out, x_p_out, start_day, n_days, l)
    Y_out = y_out

    pipe = Pipeline(steps =[("Scaler", StandardScaler()),
                             ("DM", DiffusionMaps(features=features)),
                             ("SVM", SVC(C=10, gamma=1, kernel="rbf"))])
    pipe.fit(X_train, Y_train)

    Y_train_p = pipe.predict(X_train)
    Y_out_p = pipe.predict(X_out)

    rnd_guess = DummyClassifier(strategy="uniform")
    rnd_guess.fit(X_train, Y_train)

    Y_train_rg = rnd_guess.predict(X_train)
    Y_out_rg = rnd_guess.predict(X_out)

    moda_guess = DummyClassifier(strategy="most_frequent")
    moda_guess.fit(X_train, Y_train)

    Y_train_md = moda_guess.predict(X_train)
    Y_out_md = moda_guess.predict(X_out)

    ground_truth[which_state] = np.concatenate((Y_train, Y_out))
    our_prediction[which_state] = np.concatenate((Y_train_p, Y_out_p))
    rnd_prediction[which_state] = np.concatenate((Y_train_rg, Y_out_rg))
    moda_prediction[which_state] = np.concatenate((Y_train_md, Y_out_md))


split = 11 #Where to split train/out of sample test data
rnd_st_train, rnd_st_out, rnd_mc_train, rnd_mc_out = make_stats(our_prediction, rnd_prediction, ground_truth, split, "random_guess")
moda_st_train, moda_st_out, moda_mc_train, moda_mc_out = make_stats(our_prediction, moda_prediction, ground_truth, split, "moda")

stats_data = pd.DataFrame(columns=["tag" , "set", "sample-count", "type", "pvalue", "statistic"])
stats_data = stats_data.append(rnd_st_train, ignore_index=True)
stats_data = stats_data.append(rnd_st_out, ignore_index=True)
stats_data = stats_data.append(rnd_mc_train, ignore_index=True)
stats_data = stats_data.append(rnd_mc_out, ignore_index=True)
stats_data = stats_data.append(moda_st_train, ignore_index=True)
stats_data = stats_data.append(moda_st_out, ignore_index=True)
stats_data = stats_data.append(moda_mc_train, ignore_index=True)
stats_data = stats_data.append(moda_mc_out, ignore_index=True)
stats_data.to_csv(stats_file, index=False)
######################################
# End of Stats Generation
######################################
print("Done!")
