
#REVISION NOTES:
## 01-18-2018: created this version. (previous version was 'MxNx1 Neural network-Analysis 11292017')
## 03-13-2018: added the pima indian diabetes dataset to the data_set_select function

# %%
#------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import time
from io import StringIO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_moons,make_circles
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from matplotlib.colors import ListedColormap
import matplotlib.mlab as mlab
plt.style.use('seaborn-white')
import math
from sklearn.metrics import confusion_matrix
import sys
import os


# %%
#-------------------------------------------------------------------------------------
def my_histogram (hist_data):
    """Histogram plot of the data"""
    plt.hist(hist_data[0:len(hist_data)],bins=50,stacked=False)
    plt.legend()
    plt.show()
 # %%   

def my_scattermatrix(scat_data):
    """scatter matrix plot of the dataset"""
    scatter_matrix(scat_data,alpha=0.2,diagonal='hist')
# %%

def sigmoid(x,deriv=False):
    """sigmoid activation function and its derivative"""
    if(deriv==True):
        #returns the derivative of the activation function
        return x * (1 - x)
    
        #returns the activation function
    return 1/(1+np.exp(-x))

    
def ReLu(b,deriv=False):
    """ReLu activation function and its derivative"""
    
    if(deriv==True):
        gprime=[]
        for k in range(b.shape[0]):
            if (b[k]==0):
                gprime.append(0)
            else:
                gprime.append(1)
        return np.asarray(gprime)
        
    #return Relu activation function
    return np.maximum(b,0)

def tanh(x,deriv=False):
    """tanh activation function and its derivative"""
    if(deriv==True):
        #returns the derivative of the activation function
        return 1-(np.tanh(x))**2
    
        #returns the activation function
    return np.tanh(x)

'************************************************************************'

# %%
'*********************************************************************'
def result_scatter(the_test_results,the_actual_output):
    xx_square=[]
    yy_square=[]
    predict_square=[]
    xx_dot=[]
    yy_dot=[]
    predict_dot=[]

##    df=pd.DataFrame(the_test_results)
    cm=plt.cm.get_cmap('RdYlBu_r')
##    print(the_actual_output)
##    print(the_actual_output.shape[0])
    
    for i in range(the_actual_output.shape[0]):
        if(the_actual_output[i][0]==1):
            xx_square.append(the_test_results[i][0])
            yy_square.append(the_test_results[i][1])
            predict_square.append(the_test_results[i][2])
##            print(xx_square)
##            print(yy_square)
##            print(predict_square) 
        else:
            xx_dot.append(the_test_results[i][0])
            yy_dot.append(the_test_results[i][1])
            predict_dot.append(the_test_results[i][2])
            
    plt.scatter(xx_square,yy_square,vmin=0,vmax=1,cmap=cm,c=predict_square,marker='s')
    plt.scatter(xx_dot,yy_dot,vmin=0,vmax=1,cmap=cm,c=predict_dot,marker='o')
    
    #s=plt.scatter(df[0],df[1],vmin=0,vmax=1,cmap=cm,c=df[2])
    
    plt.colorbar()
    plt.show()

'*********************************************************************'
def mash_it(test_data_inputs):
    x=test_data_inputs[:,0]
    y=test_data_inputs[:,1]

    #test data inputs meshed 
    X,Y=np.meshgrid(x,y)

    #meshed inputs reshaped to feed onto the model
    X=X.reshape(X.size,1)
    Y=Y.reshape(Y.size,1)
    #meshed inputs combined into an array
    V=np.column_stack((X,Y))
##    print (V.shape)
##    print(V)
    
    return V
'*********************************************************************'

def my_contour_plot(the_data,test_data_shape):
    s=test_data_shape.shape[0]
    X=the_data[:,0].reshape(s,s)
    Y=the_data[:,1].reshape(s,s)
    Z=the_data[:,2].reshape(s,s)

    print('X:\n',X)
    print('Y:\n',Y)
    print('Z:\n',Z)
    
    plt.figure()
    cp=plt.contourf(X,Y,Z,160,vmin=0,vmax=1,cmap='RdYlBu_r',c=Z)
    plt.colorbar(cp)
    plt.show()
'*********************************************************************'
def plot_training_data(X,y):
    xx_r=[]
    yy_r=[]
    xx_b=[]
    yy_b=[]
    plt.figure(figsize=(15,5))
    plt.title("Classification Training Set")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(b=True, which='major', color='b', linestyle='-')
    for t in range(X.shape[0]):
        if (y[t][0]==1):
            xx_r.append(X[t][1])
            yy_r.append(X[t][2])
            plt.scatter(x=xx_r, y=yy_r,color='r')
        else:
            xx_b.append(X[t][1])
            yy_b.append(X[t][2])
            plt.scatter(x=xx_b, y=yy_b,color='b')
    plt.show()
    #return (xx_r,yy_r,xx_b,yy_b)

   # %% 
#---------------------------------------------------------------------------
def training_algorithm(X,y,syn0,syn1,activation_function,num_epoch,Alpha=.5):
    
    Xplot=[]
    Yplot=[]
    iteration=0
    Xplot_1=[]
    Yplot_1=[]
    syn1_plot=[]
    it=0

    # t0=time.clock
    for epoch in range(num_epoch):
        # t1=time.clock
        for i in range(X.shape[0]):   
            Z1=syn0.T.dot(X[i])
            if (activation_function=='relu'):
                A1=ReLu(Z1)
                A1=np.insert(A1,0,1, axis=0)
                gprime_1=ReLu(A1,deriv=True)
                
            elif(activation_function=='sig'):
                A1=sigmoid(Z1)
                A1=np.insert(A1,0,1, axis=0)
                gprime_1=sigmoid(A1,deriv=True)

            elif(activation_function=='tanh'):
                A1=tanh(Z1)
                A1=np.insert(A1,0,1, axis=0)
                gprime_1=tanh(A1,deriv=True)
                
            Z2=syn1.T.dot(A1)
            A2=sigmoid(Z2)
            
            The_Error=y[i,0]-A2
            
            Cost_Function=(The_Error**2)/2
              
            gprime_2=sigmoid(A2,deriv=True)
            Delta2=gprime_2*The_Error
            
            Error_1=Delta2*syn1.T

            Delta1=gprime_1*Error_1
            
            #Parameter gradients for 2nd layer
            #element wise multiplication of the delta 2 and the syn1 weight matrix
            gradients_2=np.outer(Delta2,A1)
            syn1=syn1+(Alpha*gradients_2.T)
            
            #Parameter gradients for 1st layer
            gradients_1=np.outer(Delta1,X[i])
            
            #print (Delta1,gradients_1)
            
            syn0=syn0+(Alpha*gradients_1[1:].T)
            
            it=it+1
            if(i % (1000))==0:
                Xplot_1.append(it)
                Yplot_1.append(Cost_Function)
                #print('epoch: ',epoch)
        iteration=iteration+1
        Xplot.append(iteration)
        Yplot.append(Cost_Function)
        syn1_plot.append(A2)
##    print ('syn0 trained:\n',syn0)
##    print("---------")
##    print ('syn1 trained:\n',syn1)
##    print("---------")

##    plot_cost_function(Xplot,Yplot)
##    plot_iterative_cost_function(Xplot_1,Yplot_1)
##    plot_training_output(Xplot,syn1_plot)
    
    return syn0,syn1,Xplot,Yplot,Cost_Function[0]

    #print("Epoch:%d, Cost: %.8f, Time: %.4f"%(epoch, Cost_Function,time.clock()-t0))
    #print("time:",time.clock()-t0)
    #print("SGD Elapsed Time:",time.clock()-t1)

  # %%  
#--------------------------------------------------------------------------------------
def test_the_model(test_data,syn0,syn1,activation_function,X,y,test_output):
    X1_instance=[]
    X2_instance=[]
    y_model_output=[]
    my_results=[]
    plot_color=[]
    raw_predictions=[]
    predictions=[]
    binary_pred=[]
    
    for p in range(test_data.shape[0]):
        TD=np.insert(test_data[p],0,1, axis=0)
        #print(TD)
        z1=syn0.T.dot(TD)

        if (activation_function=='relu'):
            a1=ReLu(z1)
            
        elif(activation_function=='sig'):
            a1=sigmoid(z1)

        elif(activation_function=='tanh'):
            a1=tanh(z1)
    
        a1=np.insert(a1,0,1, axis=0)
    
        z2=syn1.T.dot(a1)
        a2=sigmoid(z2)
    
        #list used for scatter plotting 2 inputs (features) only
        X1_instance.append(TD[1])
        X2_instance.append(TD[2])
        y_model_output.append(a2[0])

        #setting the color based on the output
        if (a2[0])<.5:
            the_color='b'
            bp=0
        else:
            the_color='r'
            bp=1
        
        plot_color.append(the_color)

        #list of the output predictions
        predictions.append(a2[0])
        binary_pred.append(bp)
        #raw_predictions.append(a2[0])
    
    #print('Predictions:',predictions)

    predict=np.asarray(predictions)
    raw=np.asarray(raw_predictions)
    r=raw.reshape(len(raw_predictions),1)

    bi_pred=np.asarray(binary_pred)
    
    print('predict:\n',predict)

    the_results=np.insert(test_data,2,predict,axis=1)
    
    print('test data output\n',test_output)
    print('the_results:\n',the_results)

## CLASSIFICATION ACCURACY & CONFUSION MATRIX
    count_it=0
    TP=0
    FP=0
    FN=0
    TN=0
    target_count=the_results.shape[0]
    for g in range(the_results.shape[0]):
        if(round(predict[g])!=test_output[g]):
            count_it=count_it+1
        if(round(predict[g])==1 and test_output[g]==1):
            TP=TP+1
        if(round(predict[g])==1 and test_output[g]==0):
            FP=FP+1
        if(round(predict[g])==0 and test_output[g]==1):
            FN=FN+1
        if(round(predict[g])==0 and test_output[g]==0):
            TN=TN+1
            
        
    accur=(target_count-count_it)/target_count
    confusion_matrix=np.array([[TN,FP],[FN,TP]])

    try:
        precision_1=TP/(TP+FP)
    except ZeroDivisionError:
        precision_1=0


    recall_1=TP/(TP+FN)
    
    try:
        precision_0=TN/(TN+FN)
    except ZeroDivisionError:
        precision_0=0

    recall_0=TN/(TN+FP)

    y_round_predict=np.round(predict)
##    y_round_predict=y_round_predict.reshape(100,1)
    y_round_predict=y_round_predict.astype(int)
    
##    test_output.reshape(100,)
##    print('test output \n', test_output,'\n shape: ',test_output.shape)
##    print('predict \n', predict,'\n shape: ',predict.shape)
##    print('y_round \n', y_round_predict,'\n shape: ',y_round_predict.shape)
    
##    con_mat=confusion_matrix(test_output,y_round_predict)
    con_mat=1
    

## 

    accur_scikit=accuracy_score(test_output,bi_pred)  #***6-1-2019
    recall_scikit=recall_score(test_output,bi_pred,average=None)#***6-1-2019
    prec_scikit=precision_score(test_output,bi_pred,average=None)#***6-1-2019

    print('Model accuracy (scikit): ',accur_scikit)
    print('Model recall (scikit): ',recall_scikit)
    print('Model precision (scikit): ',prec_scikit)

##    print('Model accuracy: ',accur)


    

##    decision_boundaries(syn0,X1_instance,X2_instance,plot_color,X,y)
    
    return the_results,accur,confusion_matrix,precision_1,recall_1,precision_0,recall_0,con_mat
#-----------------------------------------------------------------------------
def decision_boundaries(syn0,X1_instance,X2_instance,plot_color,X,y):
    XX=[]
    for k in range(syn0.shape[1]):
        XX=[]
        for f in range(1,len(syn0)):
            XX.append(-syn0[0,k]/syn0[f,k])
            #print(f)
        print('Decision Boundaries:\n',XX)

        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.scatter(XX[0], 0,color='black',marker='D')
        plt.scatter(0,XX[1],color='black',marker='D')
        plt.plot((XX[0], 0),(0,XX[1]),color='black',linewidth=2)
        

    plt.scatter(X1_instance,X2_instance,color=plot_color,marker='*',s=80)

    xx_r=[]
    yy_r=[]
    xx_b=[]
    yy_b=[]
##    plt.figure(figsize=(15,5))
    plt.title("Classification Training Set")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(b=True, which='major', color='b', linestyle='-')
    for t in range(X.shape[0]):
        if (y[t][0]==1):
            xx_r.append(X[t][1])
            yy_r.append(X[t][2])
            plt.scatter(x=xx_r, y=yy_r,color='r')
        else:
            xx_b.append(X[t][1])
            yy_b.append(X[t][2])
            plt.scatter(x=xx_b, y=yy_b,color='b')
    plt.show()
    

#----------------------------------------------------------------------------

def plot_cost_function(Xplot,Yplot,num_epoch,Alpha,n_hidden,model_accuracy,activation_function):
    plt.figure(figsize=(15,5))
    plt.title('Plot Cost Function: e={}  alpha={}  n={}  accuracy={}  activation={} '.format(num_epoch,Alpha,n_hidden,model_accuracy,activation_function))
    plt.xlabel("Iterations")
    plt.ylabel("Error")
##    plt.ylim(0,1)
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.plot(Xplot, Yplot)
##    plt.show()

def plot_iterative_cost_function(Xplot_1,Yplot_1):
    plt.figure(figsize=(15,5))
    plt.title("Iterative Cost Function")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.plot(Xplot_1, Yplot_1)
    plt.show()

def plot_training_output(Xplot,syn1_plot):
    plt.figure(figsize=(15,5))
    plt.title("Training Output: last instance of training data")
    plt.xlabel("Iterations")
    plt.ylabel("Output (activation value)")
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.plot(Xplot, syn1_plot)
    plt.show()


def get_contour_data(test_data,syn0,syn1,activation_function):
    y_model_output=[]
    my_results=[]
    raw_predictions=[]
    predictions=[]                   
    for p in range(test_data.shape[0]):
        TD=np.insert(test_data[p],0,1, axis=0)
        #print(TD)
        z1=syn0.T.dot(TD)

        if (activation_function=='relu'):
            a1=ReLu(z1)
            
        elif(activation_function=='sig'):
            a1=sigmoid(z1)
            
        elif(activation_function=='tanh'):
            a1=tanh(z1)
    
        a1=np.insert(a1,0,1, axis=0)
    
        z2=syn1.T.dot(a1)
        a2=sigmoid(z2)
    
        y_model_output.append(a2[0])

        #list of the output predictions
        predictions.append(a2[0])
        
    predict=np.asarray(predictions)
    raw=np.asarray(raw_predictions)
    r=raw.reshape(len(raw_predictions),1)
    #print('predict:\n',predict)
    contour_results=np.insert(test_data,2,predict,axis=1)
    print('the_results:\n',contour_results)

    return contour_results

#----------------------------------------------------------------------------------------


def data_set_select(data_select):

    if (data_select=='ran'):
#feature matrix
        
        x1=np.random.normal(loc=0,scale=1, size=(30,1))
        x2=np.random.normal(loc=0,scale=1, size=(30,1))
        X=np.concatenate((x1,x2),axis=1)
##        my_histogram(X)
        X=np.insert(X,0,1, axis=1) #adding bias for training algorithm
        print(X)
#output matrix
        y=np.random.randint(0,2,size=(X.shape[0],1))
        print(y)
#plot the training data
##        plot_training_data(X,y)
#test data
        np.random.seed(0)
        test_data_x1=np.random.normal(loc=0,scale=1, size=(60,1))
        test_data_x2=np.random.normal(loc=0,scale=1, size=(60,1))
        test_data=np.concatenate((test_data_x1,test_data_x2),axis=1)
        print('Test Data:',test_data)
        test_output=np.random.randint(0,2,size=(test_data.shape[0],1))

#________________________CDH__________________________________   

    elif (data_select=='CDH'):
#feature matrix
        the_file=r'C:\Users\Crystal\Desktop\Programs\dataset_repo\CDH_Train.csv'
        my_data=pd.read_csv(the_file,usecols=[0,1,2])
        my_data.dropna(axis=0,inplace=True)
        my_data=my_data.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
        X_data=my_data[['Length','Width']]
        Y_data=my_data['model_target']
        
        X, test_data, y, test_output=train_test_split(X_data,Y_data)
        
        X=X.values
##        my_histogram(X)
##        my_scattermatrix(X_data)
        X=np.insert(X,0,1, axis=1)
        print(X)
#output matrix
        y=y.values.reshape(X.shape[0],1)
        
##plot the training data                            
##        plot_training_data(X,y)
#test data
        print('Test Data:',test_data.head(10))
        test_data=test_data.values
        test_output=test_output.values.reshape(test_data.shape[0],1)
        
        
#_____________________STOCK__________________________________
    elif (data_select=='stock'):
#feature matrix
        the_file=r'C:\Users\Sharyn\Desktop\Datasets\sandp500\trade_on_conditions_data\consolidated_trade_on_conditions_0.csv'
        usecols=['Slope','Slope_rate','Sigma_Spread','HL_Spread','Vol_pct','Daily_Return_ma','RSI','GL_Status']
        my_features=['Vol_pct','Slope_rate','RSI']
        my_data=pd.read_csv(the_file,usecols=usecols,nrows=10)
        my_data.dropna(axis=0,inplace=True)
        my_data['Slope_rate']=np.where(my_data['Slope_rate']>0,1,0)
        my_data['RSI']=np.where(my_data['RSI']>60,1,0)
        my_data=my_data.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
        X_data=my_data[my_features]
        Y_data=my_data['GL_Status']
        
        X, test_data, y, test_output=train_test_split(X_data,Y_data)

        X=X.values
        y=y.values.reshape(X.shape[0],1)
        
##        my_histogram(X)
##        my_scattermatrix(X_data)
##        poly=PolynomialFeatures(degree=2,include_bias=False,interaction_only=False)
##        X=poly.fit_transform(X)
##        my_scattermatrix(pd.DataFrame(X))
        X=np.insert(X,0,1, axis=1)
        print('Training Data \n',X)
        print('Training Output \n',y)
        
#plot the training data
##        plot_training_data(X,y)
#test data
        test_data=test_data.values
        test_output=test_output.values.reshape(test_data.shape[0],1)
##        test_poly=PolynomialFeatures(degree=2,include_bias=False,interaction_only=False)
##        test_data=test_poly.fit_transform(test_data)
        print('Test Data:',test_data[0:5])
        print('Test output:',test_output[0:5])
        
#_______________________MOONS__________________________________________
    elif (data_select=='moons'):
#feature matrix
        np.random.seed(0)
        X,y=make_moons(200,noise=.10)
##        my_histogram(X)
        X=np.insert(X,0,1, axis=1) #adding bias for training algorithm
        print(X)
#output matrix
        y=y.reshape(X.shape[0],1)
        print(y)
#plot the training data
##        plot_training_data(X,y)
#test data
        np.random.seed(1)
        test_data,test_output=make_moons(100,noise=.1)
        test_output=test_output.reshape(test_data.shape[0],1)
##        test_data_x1=np.random.normal(loc=0,scale=1, size=(20,1))
##        test_data_x2=np.random.normal(loc=0,scale=1, size=(20,1))
##        test_data=np.concatenate((test_data_x1,test_data_x2),axis=1)
        print('Test Data:',test_data)
##        test_output=np.random.randint(0,2,size=(test_data.shape[0],1))


#_______________________CIRCLES__________________________________________
    elif (data_select=='circles'):
#feature matrix
        np.random.seed(0)
        X,y=make_circles(200,noise=.1)
##        my_histogram(X)
        X=np.insert(X,0,1, axis=1) #adding bias for training algorithm
        print(X)
#output matrix
        y=y.reshape(X.shape[0],1)
        print(y)
#plot the training data
##        plot_training_data(X,y)
#test data
        np.random.seed(1)
        test_data,test_output=make_circles(50,noise=.1)
        test_output=test_output.reshape(test_data.shape[0],1)
##        test_data_x1=np.random.normal(loc=0,scale=1, size=(20,1))
##        test_data_x2=np.random.normal(loc=0,scale=1, size=(20,1))
##        test_data=np.concatenate((test_data_x1,test_data_x2),axis=1)
        print('Test Data:',test_data)
##        test_output=np.random.randint(0,2,size=(test_data.shape[0],1))
        

#_____________________PIMA__________________________________
    elif (data_select=='pima'):
#feature matrix
        the_file=r'C:\Users\Crystal\Desktop\Programs\dataset_repo\diabetes.csv'
        usecols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
        my_features=['BloodPressure','Age','BMI']
        my_data=pd.read_csv(the_file,usecols=usecols)

        replace_the_zeros=['Age','Glucose','BloodPressure','SkinThickness','Insulin','BMI']
        for header in replace_the_zeros:
            my_data[header]=my_data[header].replace(0,np.nan)
            mean=int(my_data[header].mean(skipna=True))
            my_data[header]=my_data[header].replace(np.nan,mean)
        
        my_data.dropna(axis=0,inplace=True)
        my_data=my_data.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
        X_data=my_data[my_features]
        Y_data=my_data['Outcome']
        
        X, test_data, y, test_output=train_test_split(X_data,Y_data)

        X=X.values
        y=y.values.reshape(X.shape[0],1)
        
##        my_histogram(X)
##        my_scattermatrix(X_data)
##        poly=PolynomialFeatures(degree=2,include_bias=False,interaction_only=False)
##        X=poly.fit_transform(X)
##        my_scattermatrix(pd.DataFrame(X))
        X=np.insert(X,0,1, axis=1)
        print('Training Data \n',X)
        print('Training Output \n',y)
        
#plot the training data
##        plot_training_data(X,y)
#test data
        test_data=test_data.values
        test_output=test_output.values.reshape(test_data.shape[0],1)
##        test_poly=PolynomialFeatures(degree=2,include_bias=False,interaction_only=False)
##        test_data=test_poly.fit_transform(test_data)
        print('Test Data:',test_data[0:5])
        print('Test output:',test_output[0:5])


    return X.shape[1],y.shape[1],X,y,test_data,test_output

#------------------------------------------------------------------------------------------------------

def decision_boundary_plot(X,y,syn0,syn1,activation_function,the_test_results,test_output,num_epoch,Alpha,n_hidden,model_accuracy):
    x1=X[:,1]
    x2=X[:,2]
    
    x1_r=[]
    x2_r=[]
    x1_b=[]
    x2_b=[]
    prob_r=[]
    prob_b=[]
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    h=.02
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min,x_max = x1.min()-1,x1.max()+1
    y_min,y_max = x2.min()-1,x2.max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    the_predict = feed_forward(np.c_[xx.ravel(),yy.ravel()],syn0,syn1,activation_function)

    # Put the result into a color plot
    Z = the_predict.reshape(xx.shape)
    plt.figure(figsize=(15,15))
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title('e={}  alpha={}  n={}  accuracy={}  activation={}'.format(num_epoch,Alpha,n_hidden,model_accuracy,activation_function))
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
    #------------------------------------------------------------------------------------
#plotting the training data (scatter)***
    xx_1=pd.DataFrame(x1).iloc[:,0]
    xx_2=pd.DataFrame(x2).iloc[:,0]
    y=pd.DataFrame(y).iloc[:,0]
##    print('xx and y: \n',xx_1,xx_2,y,list(xx_1[y==0]))
    plt.scatter(list(xx_1[y==0]),list(xx_2[y==0]),color='b')
    plt.scatter(list(xx_1[y==1]),list(xx_2[y==1]),color='r')
##    plt.scatter(list(x1[y==0]),list(x2[y==0]),color='b')
##    plt.scatter(list(x1[y==1]),list(x2[y==1]),color='r')
    
##    print('test_samples:',test_samples.Length)
    #-------------------------------------------------------------------------------------
#plotting the test data using probabilities of the predictions***
    test_samples=the_test_results
    for i in range(len(the_test_results)):
        if (test_output[i][0]==1):
            x1_r.append(test_samples[i][0])
            x2_r.append(test_samples[i][1])
            prob_r.append(test_samples[i][2])
        else:
            x1_b.append(test_samples[i][0])
            x2_b.append(test_samples[i][1])
            prob_b.append(test_samples[i][2])
    cm=plt.cm.get_cmap('RdYlBu_r')        
    plt.scatter(x1_r,x2_r,cmap=cm,vmin=0,vmax=1,c=prob_r,marker='d')
    plt.scatter(x1_b,x2_b,cmap=cm,vmin=0,vmax=1,c=prob_b,marker='d')
    
    plt.colorbar()
##    plt.show()

#-----------------------------------------------------------------------------------
def feed_forward(test_data,syn0,syn1,activation_function):
    X1_instance=[]
    X2_instance=[]
    y_model_output=[]
    my_results=[]
    plot_color=[]
    raw_predictions=[]
    predictions=[]                   
    for p in range(test_data.shape[0]):
        TD=np.insert(test_data[p],0,1, axis=0)
        #print(TD)
        z1=syn0.T.dot(TD)

        if (activation_function=='relu'):
            a1=ReLu(z1)
            
        elif(activation_function=='sig'):
            a1=sigmoid(z1)

        elif(activation_function=='tanh'):
            a1=tanh(z1)
    
        a1=np.insert(a1,0,1, axis=0)
    
        z2=syn1.T.dot(a1)
        a2=sigmoid(z2)
    
        #list used for scatter plotting 2 inputs (features) only
        X1_instance.append(TD[1])
        X2_instance.append(TD[2])
        y_model_output.append(a2[0])

        #setting the color based on the output
        if (a2[0])<.5:
            the_color='b'
        else:
            the_color='r'
        
        plot_color.append(the_color)

        #list of the output predictions
        predictions.append(round(a2[0]))
        raw_predictions.append(a2[0])
    
    #print('Predictions:',predictions)

    predict=np.asarray(predictions)
    raw=np.asarray(raw_predictions)
    r=raw.reshape(len(raw_predictions),1)
    
##    print('predict:\n',predict)

    the_results=np.insert(test_data,2,predict,axis=1)
    
    #print('test data output\n',test_output)
    #print('the_results:\n',the_results)

    return predict

# %%
def read_input_parameters():
    """read the model parameters from external text file"""
    values=[]
    with open("model_parameters.txt") as fp: 
        for line in fp: 
            a=line.split('-')
            values.append(a[0])
            print(a)
    # creating list for parameters and converting to appropiate data type

    lr=values[2].split('|')
    numIterations=values[3].split('|')
    hiddenLayerN=values[4].split('|')
    act_funct=values[5].split('|')

    if (lr[1]=='s'):
        values[2]=[float(lr[0])]
    else:
        values[2]=[float(x) for x in lr]

    if (numIterations[1]=='s'):
        values[3]=[int(numIterations[0])]
    else:
        values[3]=[int(x) for x in numIterations]

    if (hiddenLayerN[1]=='s'):
        values[4]=[int(hiddenLayerN[0])]
    else:
        values[4]=[int(x) for x in hiddenLayerN]

    if (act_funct[1]=='s'):
        values[5]=[act_funct[0]]
    else:
        values[5]=act_funct

    
    # values[3]=[int(x) for x in values[3].split('|')]
    # values[4]=[int(x) for x in values[4].split('|')]
    # values[5]=values[5].split('|')


    print(values[2])
    return values

# %%
if __name__ == "__main__":
    

    model_param=read_input_parameters()
    for z in model_param:
        print(f'{z}---{type(z)}')

    # sys.exit()


    #-----------------------------------------------------------------------------------
    # MAIN PROGRAM *****************************MAIN PROGRAM*************MAIN PROGRAM  

    startTime=time.time()

    #Data Selection (your options are: ran or CDH or stock or moons or circles or pima)
    n_inputs,n_outputs,X,y,test_data,test_output=data_set_select(data_select=model_param[0]) #model_param[0]

    #-----------------------------------------------------------------------------------

    the_score=[]
    dbp=[]
    the_learning_rate=[]
    num_interations=[]
    num_neurons=[]
    act_function=[]
    loss_function=[]

    plot_me=model_param[1] # model_param[1]

    # learn_rate=[float(model_param[2])]
    # epoch=[int(model_param[3])]
    # neurons=[int(model_param[4])]

    learn_rate=model_param[2]
    epoch=model_param[3]
    neurons=model_param[4]

    # learn_rate=[.1]
    # epoch=[100]
    # neurons=[8]

    #Hidden Layer Activation Function Select (your options are: sig or relu or tanh)
    # AF=['sig','relu','tanh']
    # AF=[model_param[5]]
    AF=model_param[5]
    # AF=['sig']

    plot_count=0



    for n_hidden,num_epoch,Alpha,activation_function in [(n_hidden,num_epoch,Alpha,activation_function) for n_hidden in neurons for num_epoch in epoch for Alpha in learn_rate for activation_function in AF]:

    #initialize layer 1 weights
        np.random.seed(0)
        syn0=np.random.normal(scale=0.1, size=(n_inputs,n_hidden))
        print('syn0:\n',syn0)

        syn1=np.random.normal(scale=0.1, size=(n_hidden+1,n_outputs))
        print('syn1:\n',syn1)
    #this matrix is already transposed
        

    #calling the training algorithm
        trained_syn0,trained_syn1,CFXplot,CFYplot,the_cost=training_algorithm(X,y,syn0,syn1,activation_function,num_epoch=num_epoch,Alpha=Alpha)

    #your trained weights
    ##    trained_syn0=trained[0]
    ##    trained_syn1=trained[1]

        print ('syn0_after:\n',trained_syn0)
        print ('syn1_after:\n',trained_syn1)

    #now test the model using your test data
        the_test_results,model_accuracy,con_matrix,precision_1,recall_1,precision_0,recall_0,con_mat=test_the_model(test_data,trained_syn0,trained_syn1,activation_function,X,y,test_output)
    ##    result_scatter(the_test_results,test_output)
        the_score.append(model_accuracy)
        the_learning_rate.append(Alpha)
        num_interations.append(num_epoch)
        num_neurons.append(n_hidden)
        act_function.append(activation_function)
        loss_function.append(the_cost)
        

        if ((n_inputs-1)==2)and plot_me==model_param[1]:  #model_param[1]
            
            dd=decision_boundary_plot(X,y,trained_syn0,trained_syn1,activation_function,the_test_results,test_output,num_epoch,Alpha,n_hidden,model_accuracy)

            dbp.append(dd)

            location=r'C:\Users\Crystal\Desktop\Programs\neural_network\NN_Result_{}.pdf'.format(plot_count)
            plt.savefig(location, dpi=None, facecolor='w', edgecolor='w',
              orientation='portrait', papertype=None, format=None,
              transparent=False, bbox_inches=None, pad_inches=0.1,
              frameon=None)
        
        cf=plot_cost_function(CFXplot,CFYplot,num_epoch,Alpha,n_hidden,model_accuracy,activation_function)

        location=r'C:\Users\Crystal\Desktop\Programs\neural_network\CF_Result_{}.pdf'.format(plot_count)
        plt.savefig(location, dpi=None, facecolor='w', edgecolor='w',
              orientation='portrait', papertype=None, format=None,
              transparent=False, bbox_inches=None, pad_inches=0.1,
              frameon=None)
        
        plot_count=plot_count+1


    ##the_mesh=mash_it(the_test_results)
    ##contour_data=get_contour_data(the_mesh,trained_syn0,trained_syn1,activation_function)
    ##my_contour_plot(contour_data,the_test_results)
        
    endTime=time.time()
    print('Duration :',endTime-startTime,' seconds')
    print('Model accuracy: ',the_score)
    print('Hidden layer size: ',num_neurons)
    print('Number of iterations: ',num_interations)
    print('Learning rate: ',the_learning_rate)
    print('Activation function: ',act_function)

    df=pd.DataFrame({'Model_accuracy':the_score,'Cost_function':loss_function,'Hidden_layer_size':num_neurons,'Number_of_iterations':num_interations,'Learning_rate':the_learning_rate,'Activation_function':act_function})

    results_location=r'C:\Users\Crystal\Desktop\Programs\neural_network\learning_table.csv'
    df.to_csv(results_location)        
    print(df)
    print('Confusion Matrix:\n',con_matrix)
    print('Precision_1:',precision_1)
    print('Recall_1:',recall_1)
    print('Precision_0:',precision_0)
    print('Recall_0:',recall_0)

    F1_0=2*((precision_0*recall_0)/(precision_0+recall_0))
    F1_1=2*((precision_1*recall_1)/(precision_1+recall_1))

    print ('F1 for 0 output:', F1_0)
    print ('F1 for 1 output:', F1_1)

    print('   \n')

    print('Confusion Matrix (SKlearn):\n',con_mat)


    ##plt.plot(CFX_data,CFY_data);plt.show()

    ##plt.show()

    ##my_histogram(X)
    ##my_scattermatrix(pd.DataFrame(X))
                  
    #--------------------------------------------------------------'



