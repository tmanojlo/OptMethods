from calculate_features import load_dataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from hmmlearn import hmm
import warnings
import matplotlib.pyplot as plt

#za hmmlearn lib
warnings.filterwarnings("ignore", category=DeprecationWarning) 

y= [0,1]

#ocekuju se wav datoteke u folderima modern i classic
X,Y = load_dataset("C:\\users\\tmanojlo\\Desktop\\music_dataset\\")
n_components = 10
average_per_component = []
for n_component in range(2,n_components):
    print("Training and testing" + str(n_component) + " components")
    kf = KFold(n_splits=10,shuffle = True)
    kf.get_n_splits(X)
    
    X = scale(X)
    #X = np.digitize(X, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    scores_list = []
    
    for train_index, test_index in kf.split(X):
        models = []
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        for i in y:
            filtered = X_train[y_train == i]
            models.append(hmm.GaussianHMM(n_components=n_component,n_iter=1000).fit(filtered))
        
        logprob = []
        for m in models:
            logprob_class = []
            for i in X_test:
                logprob_class.append(m.score([i]))
            logprob.append(logprob_class)
            
        logprob = np.array(logprob)
        y_hat = np.argmax(logprob, axis=0)
        missed = (y_hat != y_test)
        performance = (100 * (1 - np.mean(missed)))
        scores_list.append(performance)
    
        print("Average score " + str(np.mean(scores_list)))
    average_per_component.append(np.mean(scores_list))
    


plt.plot([2,3,4,5,6,7,8,9],average_per_component)
plt.title("Odnos točnosti modela i broja komponenti Gaussove mješavine")
plt.ylabel('Točnost %')
plt.xlabel('Broj komponenti Gaussove mješavine')
plt.show()




