import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap
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

def plot_region(clf, X, c=["#DDDDFF","#FFDDDD"]):

    cmap = ListedColormap(c)
    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    x2, y2 = np.meshgrid(np.arange(x_min-k, x_max+k, h), np.arange(y_min-k, y_max+k, h))

    P = clf.predict(np.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)
    plt.contourf(x2, y2, P, cmap=cmap, alpha = 1.0)
    plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
    plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)

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
 
######################################
# End of Utils
######################################

######################################
# Begin of Figure Generation
######################################


#input files
path = os.path.join("..", "data");
fig_path = os.path.join("results", "figures")
temp_file = "temp_avg.csv"
precip_file = "precip.csv"
dengue_file = "dengue.csv"
results_file = "result.csv"

results = pd.read_csv(os.path.join("..", "results", results_file))
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
    rpath = os.path.join("..", fig_path, which_state)
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
    n_rep = 200 #number of noise data repetitions
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

    ##
    # Figure Generation
    ##
    
    n_days = result["ndays"]
    l = (result["band_a"], result["band_b"])
    features = (result["feat_a"], result["feat_b"])
    start_day = result["index"]
    
    print(which_state, n_days, l, features, start_day)

    X_train = pack_data(x_t_train, x_p_train, start_day, n_days, l)
    Y_train = y_dengue

    X_out = pack_data(x_t_out, x_p_out, start_day, n_days, l)
    Y_out = y_out

    X_val = pack_data(x_t_val, x_p_val, start_day, n_days, l)
    Y_val = y_rep
    
    sc = StandardScaler()
    sc.fit(X_train)
    
    X_train_sc = sc.transform(X_train)
    X_val_sc = sc.transform(X_val)
    X_out_sc = sc.transform(X_out)

    dm = DiffusionMaps(features=features)
    dm.fit(X_train_sc)

    X_train_dm = dm.transform(X_train_sc)
    X_val_dm = dm.transform(X_val_sc)
    X_out_dm = dm.transform(X_out_sc)
    
    model = SVC(C=10, gamma=1, kernel="rbf")
    model.fit(X_train_dm, Y_train)
    print(which_state, model.score(X_train_dm, Y_train), model.score(X_val_dm, Y_val), model.score(X_out_dm, Y_out))

    light = ["#00876c", "#6eaf9a", "#b7d7cc"]
    dark = ["#d43d51", "#eb8387", "#f9c2c1"]

    fig = plt.figure(figsize=(12,10))

    c0 = dark[0]
    c1 = light[0]

    c0_light = dark[1]
    c1_light = light[1]

    val_zero = Y_val == 0
    val_one = Y_val == 1

    train_zero = Y_train == 0
    train_one = Y_train == 1

    out_zero = Y_out == 0
    out_one = Y_out == 1

    plot_region(model, X_val_dm, c=[light[2], dark[2]])

    plt.scatter(X_val_dm[val_zero,0], X_val_dm[val_zero,1], c=c1_light, s=10, label="no-outbreak(noise)")
    plt.scatter(X_val_dm[val_one,0], X_val_dm[val_one,1], c=c0_light, s=10, label="outbreak(noise)")
    plt.scatter(X_train_dm[train_zero,0], X_train_dm[train_zero,1], c=c1, s=600, marker="^", label="no-outbreak")
    plt.scatter(X_train_dm[train_one,0], X_train_dm[train_one,1], c=c0, s=600, marker="^", label="outbreak")

    plt.scatter(X_out_dm[out_zero,0], X_out_dm[out_zero,1], c=c1_light, s=600, marker="o", label="no-outbreak(out)")
    plt.scatter(X_out_dm[out_one,0], X_out_dm[out_one,1], c=c0_light, s=600, marker="o", label="outbreak(out)")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("$\phi_1$", fontsize=26)
    plt.ylabel("$\phi_2$", fontsize=26)
    plt.legend(fontsize=16)
    
    plt.title("$"+index2Date(start_day)+"$", fontsize=36, position=(0.0625,0.925))
    plt.savefig(os.path.join(rpath, "region.pdf"), orientation="landscape")
    plt.savefig(os.path.join(rpath, "region.png"))
                    
######################################
# End of Figure Generation
######################################
print("Done!")