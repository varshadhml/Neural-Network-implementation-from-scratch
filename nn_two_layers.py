from sklearn import datasets  
import numpy as np  
import matplotlib.pyplot as plt

np.random.seed(0)  
feature_set, labels = datasets.make_moons(100, noise=0.10)  
plt.figure(figsize=(10,7))  
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)

labels = labels.reshape(100, 1)

def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) *(1-sigmoid (x))

wh2 = np.random.rand(len(feature_set[0]),4)  
wh1 = np.random.rand(4,4)  
wo = np.random.rand(4, 1)  
lr = 0.5

for epoch in range(200000):  
    # feedforward
    zh2 = np.dot(feature_set, wh2)
    ah2 = sigmoid(zh2)

    zh1 = np.dot(ah2,wh1)
    ah1 = sigmoid(zh1)
    
    zo = np.dot(ah1, wo)
    ao = sigmoid(zo)

    # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - labels), 2)))
    print(error_out.sum())
    #dcost_dwo = dcost_dao * dao_dzo * dzo_dwo
    dcost_dao = ao - labels
    dao_dzo = sigmoid_der(zo) 
    dzo_dwo = ah1

    dcost_dwo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # Phase 2 =======================
   
    #dcost_dwh1 = dcost_dah1 * dah1_dzh1 * dzh1_dwh1
    #dcost_dah1 = dcost_dzo * dzo_dah1
    #dcost_dzo = dcost_dao * dao_dzo
    #dzo_dah1  = wo
    #dah1_dzh1 = sigmoid_der(zh1)     
    #dzh1_dwh1 = ah2
    
    
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah1 = wo
    dcost_dah1 = np.dot(dcost_dzo , dzo_dah1.T)
    dah1_dzh1 = sigmoid_der(zh1) 
    dzh1_dwh1 = ah2
    dcost_dwh1 = np.dot(dzh1_dwh1.T, dah1_dzh1 * dcost_dah1)

    # Phase 3 =======================
   
    #dcost_dwh2 = dcost_dah2 * dah2_dzh2 * dzh2_dwh2    
    #dcost_dah2 = dcost_dzh1 * dzh1_dah2
    #dcost_dzh1 = dcost_dah1 *dah1_dzh1
               #= (dcost_dzo * dzo_dah1) * (sigmoid_der(zh1))
    #dzh1_dah2 = wh1
    #dah2_dzh2 = sigmoid_der(zh2)     
    #dzh2_dwh2 = feature_set
 
    dcost_dzh1 = dcost_dah1 *dah1_dzh1
    dzh1_dah2 = wh1
    dcost_dah2 = np.dot(dcost_dzh1 , dzh1_dah2.T)
    dah2_dzh2 = sigmoid_der(zh2)    
    dzh2_dwh2 = feature_set
    dcost_dwh2 = np.dot(dzh2_dwh2.T, dah2_dzh2*dcost_dah2)

    # Update Weights ================
    wh2 -= lr * dcost_dwh2
    wh1 -= lr * dcost_dwh1
    wo  -= lr * dcost_dwo