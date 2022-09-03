import numpy as np
import pandas  as pd 
#numpy and pandas module
import matplotlib.pyplot as plt #Plot output
import scipy.optimize as opt #Scipy Optimization


def preprocessed_data(filepath):
    input_data=pd.read_csv(filepath)
    print("features:",input_data.columns)  
    data=input_data.loc[:,['Match', 'Date', 'Innings', 'Over', 'Runs', 'Total.Runs','Runs.Remaining','Innings.Total.Runs', 'Total.Out', 'Wickets.in.Hand','Error.In.Data']]
    #Overs remaining column added to dataframe
    data['Over_Remaining']=50-data['Over']
    #Innings 1 is considered
    data = data[data['Innings']==1]
    #considering data which have no error in them
    data=data[data["Error.In.Data"]==0]
    return data

def scorefunction(Z,data):
    L=Z[10]
    #ERROR
    MSEerror=0
    # for each wicket calculating error
    for wickets in range(1,11):
        df = data[data['Wickets.in.Hand']==wickets]
        overs =df["Over_Remaining"]
        actualruns = df['Innings.Total.Runs']-df['Total.Runs']
        predictedruns = Z[wickets-1]*np.subtract(1,np.exp(np.multiply(-1*L/Z[wickets-1],overs)))
        MSEerror+=np.sum(np.square(np.subtract(actualruns,predictedruns)))
    return MSEerror/len(data)
def DuckworthLewis(data):
    #initializing 250 as initial scores
    x = 250*np.ones(11)
    #random initial value for l-value=0.1
    x[10]=0.1
    MSE=0
    error = np.zeros(10) #Normalized loss at each wicket
    #calling minimize function from scipy optimize module
    res=opt.minimize(fun=scorefunction,x0=x,args=(data))
    #optimized Z0 values
    Z0= res.x[:10]
    #optimized L value
    L = res.x[10]
    #for each wicket calculating 
    for wickets in range(1,11):
        df = data[data['Wickets.in.Hand']==wickets]
        overs = df["Over_Remaining"]
        actualruns = df['Innings.Total.Runs']-df['Total.Runs']
        predictedruns = Z0[wickets-1]*np.subtract(1,np.exp(np.multiply(-1*L/Z0[wickets-1],overs)))
        error[wickets-1]=np.sum(np.square(np.subtract(actualruns,predictedruns)))/len(df)#Normalized error for each wicket
        MSE+=np.sum(np.square(np.subtract(actualruns,predictedruns))) #TOtal MSE 
    print(10*"*")
    print("\tMSE:",MSE/len(data))
    print(10*"*")
    return Z0,L,error,MSE

def plot10curves(Z,L):
    calculate = lambda wickets,overs : Z[wickets-1]*(1-np.exp(-1*L*overs/Z[wickets-1]))
    for wickets in range(1,11):
        x = range(0,51)
        y = np.array([calculate(wickets,overs) for overs in range(0,51)])
        plt.plot(x,y,label=str(wickets))
        plt.text(x[-5],y[-5],wickets)
        plt.xlabel("Overs Remaining")
        plt.ylabel("Percentage of resources remaining")
    plt.show()

def main():
    data=preprocessed_data("./data/04_cricket_1999to2011.csv")
    # print(data.head(10))
    Z0,L,error,MSE = DuckworthLewis(data)
    plot10curves(Z0,L)
    print("Optimal L value:",L)
    print("-"*20)
    for i in range(10):
        print(f'Z0 :{Z0[i]:.5f} for {i+1} wickets with Normalized Error: {error[i]:.5f} at an optimized L value: {L:.2f}')
    
    

if __name__=="__main__":
    main()







