import numpy as np
from sklearn.model_selection import train_test_split
from Library.ZScoreNorm import ZScoreNorm
from Library.CreateGaussRandMatrix import CreateGaussRandMatrix
from Library.sigmoid import sigmoid, d_sigmoid
from Library.tanh import tanh, d_tanh
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
# Nhat-Duc Hoang, hoangnhatduc@duytan.edu.vn
#----------------------------#----------------------------#----------------------------#----------------------------#----------------------------
class ArtificialNeuralNetworkRegression:
    def __init__(self):        
        self.W0 = np.zeros((1,1))
        self.W1 = np.zeros((1,1))
        self.M = 8
        self.alpha = 0.1
        self.MaxEp = 10        
        self.name = 'Artificial Neural Network Regression Model'
        
    def Train(self, Xtr, Ttr):   
        W0 = CreateGaussRandMatrix(self.M, D_x+1, 0, 1) # M x D_x+1  
        W1 = CreateGaussRandMatrix(1, self.M+1, 0, 1) # 1 x M+1
        for ep in range(self.MaxEp):
            if ep%10 == 0:
                print('ep = ', ep)
            for i in range(Ntr):
                # print('i = ', i)
                # forward
                t = Ttr[i]
                
                x_0 = Xtr[i,:].reshape(D_x, 1)   
                z_0 = np.vstack((x_0, 1))
                v_0 = np.matmul(W0, z_0)
                y_0 = np.zeros((v_0.shape[0], 1))
                for k in range(v_0.shape[0]):
                    y_0[k,0] = sigmoid(v_0[k,0])
                # print('v_0.shape', v_0.shape)

                z_1 = np.vstack((y_0, 1))
                # print('z_1.shape', z_1.shape)
                v_1 = np.matmul(W1, z_1)
                # print('v_1.shape', v_1.shape)
                y_1 = v_1.copy()
                y = y_1[0,0]      
                        
                # backpropagation
                e1 = t-y
                delta1 = e1 * 1
                
                e0 = np.zeros(self.M)
                delta0 = np.zeros(self.M)
                for u in range(self.M):
                    e0[u] = W1[0, u] * e1
                    delta0[u] = e0[u] * d_sigmoid(v_0[u,0])
                    
                #print('e1', e1)
                #print('delta0', delta0)
            
                for k in range(W1.shape[1]):
                    W1[0, k] = W1[0, k] + self.alpha * delta1 * z_1[k,0] 
                   
                for m in range(W0.shape[0]):            
                    for v in range(W0.shape[1]):             
                        W0[m,v] = W0[m,v] + self.alpha * delta0[m] *z_0[v,0]
                    
                # print('gW = ', gW)
        self.W0 = W0.copy()
        self.W1 = W1.copy()

    def Predict(self, Xte):
        Nte = Xte.shape[0]
        D_x = Xte.shape[1]
        W0 = self.W0.copy()
        W1 = self.W1.copy()
        Yte = np.zeros(Nte)    
        for i in range(Nte):
            # forward      
            x_0 = Xte[i,:].reshape(D_x, 1)   
            z_0 = np.vstack((x_0, 1))
            v_0 = np.matmul(W0, z_0)
            y_0 = np.zeros((v_0.shape[0], 1))
            for k in range(v_0.shape[0]):
                    y_0[k,0] = sigmoid(v_0[k,0])
            # print('v_0.shape', v_0.shape)

            z_1 = np.vstack((y_0, 1))
            # print('z_1.shape', z_1.shape)
            v_1 = np.matmul(W1, z_1)
            # print('v_1.shape', v_1.shape)
            y_1 = v_1.copy()
            y = y_1[0,0]      
            Yte[i] = y
        return Yte
    
#----------------------------
#----------------------------
def PrepareData(Data = np.genfromtxt('Datasets/GGBFS_ConcreteStrenth.csv', dtype='float', delimiter=',')):    
    D_d = Data.shape[1]
    D_x = D_d-1
    X = Data[:, 0: D_x]
    T = Data[:, -1]
    Data_z, MeanX, StdX = ZScoreNorm(Data)
    D_d = Data_z.shape[1]
    D_x = D_d-1
    Xz = Data_z[:, 0: D_x]
    Tz = Data_z[:, -1]
    Xtr, Xte, Ttr, Tte = train_test_split(Xz, Tz, test_size = 0.1, random_state=0) 
    return  Xtr, Xte, Ttr, Tte, MeanX, StdX
#----------------------------
def Compute_s(Ttr_o, Ytr_o, WeightNum):
    Ntr = Ttr_o.shape[0]
    s2 = np.sum((Ttr_o-Ytr_o)**2)/(Ntr-WeightNum)
    s = np.sqrt(s2)
    return s
#----------------------------
def compute_st(TrainingSampleNum, WeightNum, Level = 0.95):
    from scipy.stats import t        
    alpha = Level + (1 - Level)/2 # e.g., 0.975
    st = t.ppf(alpha, TrainingSampleNum - WeightNum) # e.g., 95% PI
    return st
#----------------------------
def compute_one_sample_gradient(Xi, Ti, W0, W1):
    M = W0.shape[0]
    dW0 = np.copy(W0)
    dW1 = np.copy(W1)
    D_x = Xi.shape[0]
    t = Ti            
    x_0 = Xi.reshape(D_x, 1)   
    z_0 = np.vstack((x_0, 1))
    v_0 = np.matmul(W0, z_0)
    y_0 = np.zeros((v_0.shape[0], 1))
    for k in range(v_0.shape[0]):
        y_0[k,0] = sigmoid(v_0[k,0])

    z_1 = np.vstack((y_0, 1))
    v_1 = np.matmul(W1, z_1)
    y_1 = v_1.copy()
    y = y_1[0,0]      
            
    # backpropagation
    e1 = t-y
    delta1 = e1 * 1
    
    e0 = np.zeros(M)
    delta0 = np.zeros(M)
    for u in range(M):
        e0[u] = W1[0, u] * e1
        delta0[u] = e0[u] * d_sigmoid(v_0[u,0])

    for k in range(W1.shape[1]):
        dW1[0, k] = delta1 * z_1[k,0] 
       
    for m in range(W0.shape[0]):            
        for v in range(W0.shape[1]):             
            dW0[m,v] = delta0[m] *z_0[v,0]
    dW = np.append(dW0, dW1)            
    return dW
#----------------------------
def Compute_FtFinv(Xtr, Ttr, W0, W1):
    Ntr = Xtr.shape[0]
    RowSize = W0.size + W1.size
    F = np.zeros((Ntr, RowSize))
    for i in range(Ntr):
        x = Xtr[i, :]
        t = Ttr[i] 
        dW_x = compute_one_sample_gradient(x, t, W0, W1)
        F[i,:] = np.copy(dW_x)
    Ft = F.transpose()
    FtF = np.matmul(Ft, F)
    FtFinv = np.linalg.pinv(FtF)
    return FtFinv   
#----------------------------
def Compute_Bound(Xtr, Ttr, FtFinv, s, st, W0, W1, Level = 0.95):
    RowSize = W0.size + W1.size
    Ntr = Xtr.shape[0]
    D_x = Xtr.shape[1]
    Bounds = np.zeros(Ntr)
    for i in range(Ntr):
        x = Xtr[i, :]
        t = Ttr[i]
        dW_x = compute_one_sample_gradient(x, t, W0, W1)
        gi = np.copy(dW_x).reshape((RowSize,1))       
        giT = gi.transpose()       
        A = np.matmul(giT, FtFinv)
        B = np.sqrt(1 + np.matmul(A, gi))    
                
        Bound = st * s * B
        Bounds[i] = Bound
    return Bounds
#----------------------------
def ComputeRmse(Y, Yp):
    msef = np.sum((Y-Yp)**2, 0)/Y.shape[0]
    f = np.sqrt(msef)
    return f
#----------------------------
#----------------------------
Data = np.genfromtxt('Datasets/GGBFS_ConcreteStrenth.csv', dtype='float', delimiter=',')
D_x = Data.shape[1] - 1
Xtr, Xte, Ttr, Tte, MeanX, StdX = PrepareData(Data)
Ntr = Xtr.shape[0]
Nte = Xte.shape[0]

# general delta rule: w = w + alpha*Delta*z
# z = input to the neuron

Set_Level = 0.9 # Level of confidence

Annr = ArtificialNeuralNetworkRegression()
Annr.M = 8
Annr.alpha = 0.1
Annr.MaxEp = 100

Annr.Train(Xtr, Ttr)
Ytr = Annr.Predict(Xtr)
Yte = Annr.Predict(Xte)

Ttr_o = Ttr*StdX[-1] + MeanX[-1]
Ytr_o = Ytr*StdX[-1] + MeanX[-1]

Tte_o = Tte*StdX[-1] + MeanX[-1]
Yte_o = Yte*StdX[-1] + MeanX[-1]
#-----------------------------------------------------------
WeightNum = Annr.W0.size + Annr.W1.size

s = Compute_s(Ttr_o, Ytr_o, WeightNum)

TrainingSampleNum = Xtr.shape[0]

st = compute_st(TrainingSampleNum, WeightNum, Level = Set_Level)
#-----------------------------------------------------------
FtFinv = Compute_FtFinv(Xtr, Ttr, Annr.W0, Annr.W1)

Bound_tr = Compute_Bound(Xtr, Ttr, FtFinv, s, st, Annr.W0, Annr.W1, Level = Set_Level)
Bound_te = Compute_Bound(Xte, Tte, FtFinv, s, st, Annr.W0, Annr.W1, Level = Set_Level)

Count_tr = 0
for i in range(Ntr):
    if Ttr_o[i] >= Ytr_o[i] - Bound_tr[i] and Ttr_o[i] <= Ytr_o[i] + Bound_tr[i]:
        Count_tr = Count_tr + 1

Prop_train = Count_tr*100/Ntr
print('Prop in intervals = ', Prop_train)

Count_te = 0
for i in range(Nte):
    if Tte_o[i] >= Yte_o[i] - Bound_te[i] and Tte_o[i] <= Yte_o[i] + Bound_te[i]:
        Count_te = Count_te + 1

Prop_test = Count_te*100/Nte
print('Prop in intervals = ', Prop_test)

results = np.hstack((Tte_o.reshape((Nte,1)), (Yte_o-Bound_te).reshape((Nte,1)), (Yte_o+Bound_te).reshape((Nte,1))))
np.savetxt("results.csv", results, fmt = '%10.5f', delimiter=",")

PI_Width_Train = 2*Bound_tr
PI_Width_Test = 2*Bound_te

Mean_PI_Width_Train = np.mean(PI_Width_Train)
Mean_PI_Width_Test = np.mean(PI_Width_Test)

print('Mean_PI_Width_Train', Mean_PI_Width_Train)
print('Mean_PI_Width_Test', Mean_PI_Width_Test)

#-----------------------------------------------------------
RMSE_train = ComputeRmse(Ttr_o, Ytr_o)
print('RMSE_train = ', RMSE_train)

MAPE_train = mean_absolute_percentage_error(Ttr_o, Ytr_o)
print('MAPE_train = ', MAPE_train*100)

R2_train = r2_score(Ttr_o, Ytr_o)
print('R2_train = ', R2_train)

# --

RMSE_test= ComputeRmse(Tte_o, Yte_o)
print('RMSE_test = ', RMSE_test)

MAPE_test = mean_absolute_percentage_error(Tte_o, Yte_o)
print('MAPE_test = ', MAPE_test*100)

R2_test = r2_score(Tte_o, Yte_o)
print('R2_test = ', R2_test)
#=========
label_str = str(Set_Level*100) + '%PI'

fig = plt.figure()
matplotlib.rc('font', size=14) # adjust font size

X_axis = range(Ntr)
plt.plot(X_axis, Ttr_o, 'o-', color ='blue', linewidth = 2, label = 'Actual')
plt.plot(X_axis, Ytr_o, 's--', color ='red', linewidth = 2, label = 'Predicted')
plt.fill_between(X_axis, Ytr_o-Bound_tr, Ytr_o+Bound_tr, alpha=0.2, label = label_str)

plt.xlabel('Data sample')
plt.ylabel('Variable')
plt.title('Result')
plt.legend()
plt.grid()
#=========
fig = plt.figure()
matplotlib.rc('font', size=14) # adjust font size

X_axis = range(Nte)
plt.plot(X_axis, Tte_o, 'o-', color ='blue', linewidth = 2, label = 'Actual')
plt.plot(X_axis, Yte_o, 's--', color ='red', linewidth = 2, label = 'Predicted')
plt.fill_between(X_axis, Yte_o-Bound_te, Yte_o+Bound_te, alpha=0.2, label = label_str)

plt.xlabel('Data sample')
plt.ylabel('Variable')
plt.title('Result')
plt.legend()
plt.grid()
#=========
plt.show()  
    
   
    











       



