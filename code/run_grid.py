import pandas as pd
from glob import glob
import numpy as np
import os
import sys
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks
import warnings
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import timeit
import csv
import time
from scipy.stats import entropy
import re
warnings.filterwarnings("ignore")
matplotlib.use("Agg")

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


def make_grid(x_t_train, x_t_val, x_t_test, x_p_train, x_p_val, x_p_test, y_d_train, y_d_val, y_d_test, n_days, l, model):
    
    years, lengths, days = x_t_test.shape
    
    m_train = {}
    m_val = {}
    m_test = {}
    
    m_train['accuracy'] = np.zeros(days - n_days)
    m_val['accuracy'] = np.zeros(days - n_days)
    m_test['accuracy'] = np.zeros(days - n_days)
    
    m_train['f1'] = np.zeros(days - n_days)
    m_val['f1'] = np.zeros(days - n_days)
    m_test['f1'] = np.zeros(days - n_days)
    
    m_train['recall'] = np.zeros(days - n_days)
    m_val['recall'] = np.zeros(days - n_days)
    m_test['recall'] = np.zeros(days - n_days)
    
    m_train['precision'] = np.zeros(days - n_days)
    m_val['precision'] = np.zeros(days - n_days)
    m_test['precision'] = np.zeros(days - n_days)

    for start in range(0, days - n_days):
        X_train = pack_data(x_t_train, x_p_train, start, n_days, l)
        Y_train = y_d_train
        
        X_val = pack_data(x_t_val, x_p_val, start, n_days, l)
        Y_val = y_d_val

        X_test = pack_data(x_t_test, x_p_test, start, n_days, l)
        Y_test= y_d_test

        model.fit(X_train, Y_train) 

        y_p_train = model.predict(X_train)
        y_p_val = model.predict(X_val)
        y_p_test = model.predict(X_test)
        
        m_train['accuracy'][start] = accuracy_score(Y_train, y_p_train)
        m_val['accuracy'][start] = accuracy_score(Y_val, y_p_val)
        m_test['accuracy'][start] = accuracy_score(Y_test, y_p_test)
        
        m_train['f1'][start] = f1_score(Y_train, y_p_train)
        m_val['f1'][start] = f1_score(Y_val, y_p_val)
        m_test['f1'][start] = f1_score(Y_test, y_p_test)
        
        m_train['recall'][start] = recall_score(Y_train, y_p_train)
        m_val['recall'][start] = recall_score(Y_val, y_p_val)
        m_test['recall'][start] = recall_score(Y_test, y_p_test)
        
        m_train['precision'][start] = precision_score(Y_train, y_p_train)
        m_val['precision'][start] = precision_score(Y_val, y_p_val)
        m_test['precision'][start] = precision_score(Y_test, y_p_test)
        
    return m_test, m_val, m_train

def progress(count, total, suffix=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total+sys.float_info.epsilon)))

    percents = round(100.0 * count / float(total+sys.float_info.epsilon), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben
    
def formatTime(t):
    s = t
    m = int(s/60)
    h = int(m/60)
    s = int(math.fmod(s, 60))
    m = int(math.fmod(m, 60))
    return str(h)+":"+str(m).zfill(2)+":"+str(s).zfill(2)

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

def index2Date(index):
    days=(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    month=("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    
    xticks=np.roll(days, -5)
    xticks_cum=np.cumsum(xticks)
    xtlabels = np.roll(month, -5);
    for i in range(0, len(xticks)):
        total = xticks_cum[i]
        day = xticks[i]
        mon = xtlabels[i]
        if(index < total):
            return  mon + " " + str(1 + day - (total - index));
    return "error";
    
def format_plot(ylim=True):
    days=(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    month=("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    plt.xlabel("Date", fontsize=22)
    plt.ylabel("Score", fontsize=22)
    
    xticks=np.roll(days, -4)
    xticks[0]=5
    xticks=np.cumsum(xticks)
    xtlabels = np.roll(month, -5);
    plt.xticks(xticks[0:9], xtlabels[0:9], fontsize=18)
    plt.grid()
    if(ylim):
        plt.yticks(np.arange(0, 1.01, step=0.1), fontsize=18)
        plt.ylim((0.0, 1.01))

def plot_confidence(m, s, color):
    x = np.linspace(1, len(m), len(m))
    k=3
    plt.fill_between(x, m+k*s, m-k*s, color=color+'33')
    plt.plot(x, m, color=color,linewidth=3)
    
def chi2_distance(A, B): 
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)  
                      for (a, b) in zip(A, B)]) 
  
    return chi 

def write_line(csv_file, data, name):
    writer = csv.writer(csv_file, delimiter=',')
    line = list(data)
    line.insert(0, name)
    writer.writerow(line)
######################################
# End of Utils
######################################


######################################
# Begin of Grid Model Generation
######################################


#input files
path = "data";
grid_path = os.path.join("results", "intermediate", "grid")
temp_file = "temp_avg.csv"
precip_file = "precip.csv"
dengue_file = "dengue.csv"

#Cities name
states = glob(os.path.join("..", path, "*/"))
##
#Run for each city
##
for which_state in states:
    ##
    #Read csv files
    ##
    temp = pd.read_csv(os.path.join(which_state,temp_file))
    precip = pd.read_csv(os.path.join(which_state,precip_file))
    dengue = pd.read_csv(os.path.join(which_state,dengue_file))
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
    #create output path
    ##
    state_name = which_state.split(os.sep)[2]
    rpath = os.path.join("..", grid_path, state_name)
    print("Grid search for " + state_name + "\n\n\n")
    if not os.path.exists(rpath):
        os.makedirs(rpath)
        time.sleep(1)
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
    #Generation of validation noisy data
    ##
    n_rep = 2000 #number of noise data repetitions
    precip_rep = np.repeat(precip.values, n_rep, axis=0)
    temp_rep = np.repeat(temp.values, n_rep, axis=0)
    y_rep = np.repeat(y_dengue, n_rep, axis=0)

    size = precip_rep.shape

    precip_noise = np.random.normal(0.0, 1.0, size=size)
    temp_noise = np.random.normal(0.0, 1.0, size=size)
    
    
    s = 2.5 #Noise intensity (sigma)
    x_p_val = ((precip_rep + precip_noise*s) - p_m) / p_s
    x_t_val = ((temp_rep + temp_noise*s) - t_m) / t_s
    x_t_val, x_p_val = process_data(x_t_val, x_p_val, base, n_levels)

    y_d_train = y_dengue
    y_d_out = y_out
    y_d_val = y_rep
    
    
    
    ##
    # Grid Search
    ##

    colors = {'accuracy_val': '#7777FF',
              'accuracy_test': '#FF7777', 
              'accuracy_train': '#77FF77',}

    ##
    #Grid parameters
    ##
    best =[(1, 4), (4, 7), (7, 10), (10, 13), (12, 15)] #Set of bands to use
    bdays = [3, 5, 9, 12] #Set of windows (size) to use
    nfeatures = [3] #Set of features to use (2,n), uses feature (2,3) since feature 1 is constant (Diffusion Maps)
    epsilon = ["auto"] #Epsilon for DM, automatic

    #For the set of bands
    for k in range(0, len(bdays)):
        n_days = bdays[k]
        #For the set of windows
        for j in range(0,len(best)):
            l=best[j]
            #For the set of features
            for f in range(0,len(nfeatures)):
                features=(1,nfeatures[f])
                #For the set of epsilon
                for e in range(0, len(epsilon)):
                    eps = epsilon[e]
                    fname = "ndays."+str(n_days)+"_range."+str(l)+"_feat."+str(features)+"_eps."+str(eps)
                    print("N Days = " + str(n_days) + " Range = " + str(l) + " Features = " + str(features) + " Epsilon = " + str(eps))

                    model = Pipeline(steps =[("Scaler", StandardScaler()),
                                             ("DM", DiffusionMaps(features=features)),
                                             ("SVM", SVC(C=10, gamma=1, kernel="rbf"))]) 

                    start_time = timeit.default_timer()

                    m_test_, m_val_, m_train_ = make_grid(x_t_train, x_t_val, x_t_out, x_p_train, x_p_val, x_p_out, y_d_train, y_d_val, y_d_out, n_days, l,  model)
                    
                    elapsed = timeit.default_timer() - start_time
                    progress(s, s, " Total time: " + formatTime(elapsed))


                    with open(os.path.join(rpath, fname+".csv"), "w") as csv_file:
                        writer = csv.writer(csv_file, delimiter=',')
                        for key in m_test_:
                            line = list(m_test_[key])
                            line.insert(0, str(key)+"_test")
                            writer.writerow(line)
                        for key in m_train_:
                            line = list(m_train_[key])
                            line.insert(0, str(key)+"_train")
                            writer.writerow(line)
                        for key in m_val_:
                            line = list(m_val_[key])
                            line.insert(0, str(key)+"_val")
                            writer.writerow(line)


                    fig = plt.figure(figsize=(20,10))
                    acc = m_test_['accuracy']
                    acc2 = m_train_['accuracy']
                    acc3 = m_val_['accuracy']
                    print("\n count: " + str(len(acc3[acc3 > 0.8])) + 
                             " mean acc: " + str(np.mean(acc3[acc3 > 0.8])) +
                              " max acc: " + str(max(acc3)))

                    plt.plot(acc, color=colors["accuracy_test"], linewidth=4)
                    plt.plot(acc2, color=colors["accuracy_train"], linewidth=4)
                    plt.plot(acc3, color=colors["accuracy_val"], linewidth=4)
                    plt.legend(["Test acc.", "Train acc.", "Val. acc."], fontsize=22)
                    format_plot()
                    plt.show()
                    fig.savefig(os.path.join(rpath, fname+".png"))
                    plt.close()
                    
                    
######################################
# End of Grid Model Generation
######################################

######################################
# Begin of Smoothing and Functionals
######################################

colors = {'accuracy_val': '#7777FF',
          'accuracy_test': '#FF7777', 
          'accuracy_train': '#77FF77',}



base_size = 15 #Smooth (convolution) filter size
mean_filter = np.ones((base_size,)) * 1/base_size
smoothed_path = os.path.join("results", "intermediate", "smoothed")

#For each city
for state in states:
    state_name = state.split(os.sep)[2]
    dirname = os.path.join("..", smoothed_path, state_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        time.sleep(1)
        
    #For each grid result
    for file in glob(os.path.join("..", grid_path, state_name, "*.csv")):
        ##
        #Read data
        ##
        data = pd.read_csv(file, header=None, index_col=[0]).transpose()
        filename = file.split(os.sep)[-1]

        acc_test = data["accuracy_test"].values
        acc_val = data["accuracy_val"].values
        acc_train = data["accuracy_train"].values

        ##
        #Smooth data
        ##
        mean_acc_test = np.convolve(acc_test, mean_filter, mode="same")
        mean_acc_val = np.convolve(acc_val, mean_filter, mode="same")
        mean_acc_train = np.convolve(acc_train, mean_filter, mode="same")

        acc_test_pad = np.pad(acc_test, base_size//2, mode="reflect")
        acc_val_pad = np.pad(acc_val, base_size//2, mode="reflect")
        acc_train_pad = np.pad(acc_train, base_size//2, mode="reflect")

        mean_acc_test_pad = np.pad(acc_test, base_size//2, mode="reflect")
        mean_acc_val_pad = np.pad(acc_val, base_size//2, mode="reflect")
        mean_acc_train_pad = np.pad(acc_train, base_size//2, mode="reflect")

        ##
        #Calculate some other properties for each grid model
        ##
        vsize = len(mean_acc_test)
        un_test = np.zeros((vsize,))
        un_val = np.zeros((vsize,))
        un_train = np.zeros((vsize,))

        div_test = np.zeros((vsize,))
        div_val = np.zeros((vsize,))

        chi2_test = np.zeros((vsize,))
        chi2_val = np.zeros((vsize,))

        for i in range(0, vsize):
            un_test[i] = np.mean((acc_test_pad[i:(i+base_size)] - mean_acc_test[i])**2)
            un_val[i] = np.mean((acc_val_pad[i:(i+base_size)] - mean_acc_val[i])**2)
            un_train[i] = np.mean((acc_train_pad[i:(i+base_size)] - mean_acc_train[i])**2)

            div_test[i] = entropy(mean_acc_train_pad[i:(i+base_size)], mean_acc_test_pad[i:(i+base_size)])
            div_val[i] = entropy(mean_acc_train_pad[i:(i+base_size)], mean_acc_val_pad[i:(i+base_size)])

            chi2_test[i] = chi2_distance(mean_acc_train_pad[i:(i+base_size)], mean_acc_test_pad[i:(i+base_size)])
            chi2_val[i] = chi2_distance(mean_acc_train_pad[i:(i+base_size)], mean_acc_val_pad[i:(i+base_size)])

        un_test = np.sqrt(un_test)
        un_val = np.sqrt(un_val)
        un_train = np.sqrt(un_train)



        ##
        #Save results
        ##
        with open(os.path.join("..", smoothed_path, state_name, filename), "w") as csv_file:
            write_line(csv_file, mean_acc_test, "mean_accuracy_test")
            write_line(csv_file, mean_acc_val, "mean_accuracy_val")
            write_line(csv_file, mean_acc_train, "mean_accuracy_train")
            write_line(csv_file, acc_test, "accuracy_test")
            write_line(csv_file, acc_val, "accuracy_val")
            write_line(csv_file, acc_train, "accuracy_train")
            write_line(csv_file, un_test, "uncertainty_test")
            write_line(csv_file, un_val, "uncertainty_val")
            write_line(csv_file, un_train, "uncertainty_train")
            write_line(csv_file, div_val, "divergence_val")
            write_line(csv_file, div_test, "divergence_test")
            write_line(csv_file, chi2_val, "chi2_val")
            write_line(csv_file, chi2_test, "chi2_test")



        print(file)
        fig = plt.figure(figsize=(20,10))
        plot_confidence(mean_acc_test, un_test, color=colors["accuracy_test"])
        plot_confidence(mean_acc_train, un_train, color=colors["accuracy_train"])
        plot_confidence(mean_acc_val, un_val, color=colors["accuracy_val"])
        plt.legend(["Test acc.", "Train acc.", "Val acc."])
        format_plot()
        plt.show()
        fig.savefig(os.path.join("..", smoothed_path, state_name, filename[:-3]+"png"))
        plt.close()

######################################
# End of Smoothing and Functionals
######################################

######################################
# Begin of Result Selection
######################################

base_size = 10 #distance between peaks to find in the data
rmax = []
rmean = []
selection_path = os.path.join("results", "intermediate", "selection")

result_data = pd.DataFrame(columns=["city", "max_acc_test", "mean_acc_test", 
                                       "max_acc_train", "mean_acc_train", 
                                       "max_acc_val", "mean_acc_val",
                                       "uncertainty", "divergence", "diff", "chi2",
                                       "index", "date", "file", "sum",
                                       "ndays", "band_a", "band_b", "feat_a", "feat_b", "eps"])

fname_pattern = re.compile('ndays\.(\d+)\_range\.\((\d+),\s(\d+)\)\_feat\.\((\d+),\s(\d+)\)\_eps\.(\w+)\.csv')
#For each city
for state in states:
    state_name = state.split(os.sep)[2]
    mean_score = pd.DataFrame(columns=["max_acc_test", "mean_acc_test", 
                                       "max_acc_train", "mean_acc_train", 
                                       "max_acc_val", "mean_acc_val",
                                       "uncertainty", "divergence", "diff", "chi2",
                                       "index", "date", "file"])
    max_score = pd.DataFrame(columns=["max_acc_test", "mean_acc_test", 
                                      "max_acc_train", "mean_acc_train", 
                                      "max_acc_val", "mean_acc_val",
                                      "uncertainty", "divergence", "diff", "chi2",
                                      "index", "date", "file"])


    #For each grid model
    for file in glob(os.path.join("..", smoothed_path, state_name, "*.csv")):
        data = pd.read_csv(file, header=None, index_col=[0]).transpose()

        ##
        #Read the data
        ##
        acc_test = data["accuracy_test"].values
        acc_val = data["accuracy_val"].values
        acc_train = data["accuracy_train"].values
        uncertainty = data["uncertainty_val"].values
        divergence = data["divergence_val"].values * 100
        chi2 = data["chi2_val"].values
        mean_acc_test = data["mean_accuracy_test"].values
        mean_acc_val = data["mean_accuracy_val"].values
        mean_acc_train = data["mean_accuracy_train"].values

        ##
        #Find peaks based on mean accuracy
        ##
        idx_mean, _ = find_peaks(mean_acc_val, distance=base_size//2)
        for i_mean in idx_mean:
            mean_score = mean_score.append({  "max_acc_test": acc_test[i_mean], 
                                   "mean_acc_test": mean_acc_test[i_mean],
                                   "max_acc_train": acc_train[i_mean], 
                                   "mean_acc_train": mean_acc_train[i_mean],
                                   "max_acc_val": acc_val[i_mean], 
                                   "mean_acc_val": mean_acc_val[i_mean],
                                   "uncertainty": uncertainty[i_mean],
                                   "divergence": divergence[i_mean],
                                   "diff": acc_train[i_mean]-acc_val[i_mean],
                                   "chi2": chi2[i_mean],
                                   "index": i_mean, 
                                   "date": index2Date(i_mean),
                                   "file": file.split(os.sep)[-1]}, ignore_index=True)
        ##
        #Find peaks based on accuracy
        ##
        idx_max, _ = find_peaks(acc_val, distance=base_size//2)
        for i_max in idx_max:
            max_score = mean_score.append({  "max_acc_test": acc_test[i_max], 
                                   "mean_acc_test": mean_acc_test[i_max],
                                   "max_acc_train": acc_train[i_max], 
                                   "mean_acc_train": mean_acc_train[i_max],
                                   "max_acc_val": acc_val[i_max], 
                                   "mean_acc_val": mean_acc_val[i_max],
                                   "uncertainty": uncertainty[i_max],
                                   "divergence": divergence[i_max],
                                   "diff": acc_train[i_max]-acc_val[i_max],
                                   "chi2": chi2[i_max],
                                   "index": i_max, 
                                   "date": index2Date(i_max),
                                   "file": file.split(os.sep)[-1]}, ignore_index=True)

    max_score["sum"] = (max_score["mean_acc_val"]+max_score["mean_acc_train"])/2
    max_score = max_score.sort_values(["sum"], ascending=False)
    
    mean_score["sum"] = (mean_score["mean_acc_val"]+mean_score["mean_acc_train"])/2
    mean_score = mean_score.sort_values(["sum"], ascending=False)
    
    rmax.append(max_score)
    rmean.append(mean_score)
    if(max_score.size > 0):
        line = max_score.iloc[0];
        values = fname_pattern.match(line['file']).groups()
        line["city"] = state_name
        line["ndays"] = values[0]
        line["band_a"] = values[1]
        line["band_b"] = values[2]
        line["feat_a"] = values[3]
        line["feat_b"] = values[4]
        line["eps"] = values[5]
        result_data = result_data.append(line)
        print("\n\n"+state_name+"\n\n")
        print(line)

    dirname = os.path.join("..", selection_path, state_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        time.sleep(1)

    max_score.to_csv(os.path.join(dirname,"max_"+str(base_size)+".csv"), float_format="%.2f", index=False)
    mean_score.to_csv(os.path.join(dirname,"mean_"+str(base_size)+".csv"), float_format="%.2f", index=False)

######################################
# End of Result Selection
######################################

######################################
# Begin Save Final Result
######################################

result_path = "results"
dirname = os.path.join("..", result_path)
if not os.path.exists(dirname):
    os.makedirs(dirname)
    time.sleep(1)
result_data.to_csv(os.path.join(dirname, "result.csv"), float_format="%.2f", index=False)
######################################
# End Save Final Result
######################################
print("Done!")