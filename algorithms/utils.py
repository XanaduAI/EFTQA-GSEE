import numpy as np
import ruptures as rpt
import scipy
from trendfilter.trendfilter import trend_filter


def filter_(acdf):
    acdf_filter_ =  trend_filter(np.arange(len(acdf)), acdf.real , l_norm=1,
            alpha_1=10,alpha_2=0,sigma = 100,
            constrain_zero=True, monotonic=True, positive = True)

    algo = rpt.KernelCPD().fit(acdf_filter_['y_fit'])
    cps = [0] + algo.predict(1)
    cps[-1] += 1

    return cps[1]



def find_first_jump_left(acdf,alpha = 0.01):
    cut = 50
    bkp = 0
    while cut<len(acdf):
        acdf_trial = acdf[:cut]

        algo = rpt.KernelCPD(min_size=10).fit(acdf_trial)
        trial = algo.predict(1)[0]


        F = check_stat_relevant(acdf_trial,trial)

        df = len(acdf_trial)-2
        p_value = scipy.stats.f.cdf(F, 1, df)
        if p_value > alpha:
            mean1 = np.mean(acdf_trial[:trial])
            mean2 = np.mean(acdf_trial[trial:])
            std1  = np.std(acdf_trial[:trial])
            perc = np.percentile(acdf_trial[trial:],25)
            if mean2>mean1:
                if perc>0+2*std1:
                    bkp = trial
                    break
        cut += 25
    if bkp ==0:
        bkp = len(acdf)-1
        print('not converged')
    return bkp

def find_first_jump_right(acdf,alpha = 0.01):
    cps = [len(acdf)]
    for i in range(100):

        acdf_trial = acdf[:cps[-1]]
#
        algo = rpt.KernelCPD().fit(acdf_trial)
        trial = algo.predict(1)[0]


        F = check_stat_relevant(acdf_trial,trial)
        df = len(acdf_trial)-2
        p_value = scipy.stats.f.cdf(F, 1, df)
    #     print(p_value)
        if p_value < 1-alpha:
            break
        if acdf_trial[:trial].mean() > acdf_trial[trial:].mean():
            break

        if np.mean(acdf_trial[:trial])<0:
            break
        cps.append(trial)
    return cps[-1]


def check_stat_relevant(acdf, bkp):

    SSbet  = len(acdf[:bkp]) * (np.mean(acdf[:bkp]) - acdf.mean())**2
    SSbet += len(acdf[bkp:]) * (np.mean(acdf[bkp:]) - acdf.mean())**2

    MSbetween = SSbet/1

    SSwithin  = np.sum((acdf[:bkp]-acdf[:bkp].mean()) **2)
    SSwithin += np.sum((acdf[bkp:]-acdf[bkp:].mean())**2)

    MSwithin = SSwithin/(len(acdf)-2)
    F = MSbetween/MSwithin


    return float(F)


def find_large_variance(acdf,v,s=3):
    guess = []
    for sigma in range(3,4):
        for s in range(30,35,5):
            for r in range(8,9):
                a = large_variance(acdf,v,sigma,size = (r*s/10,s))
                guess.append(int(a))
    return int(np.percentile(guess,25))

def large_variance(acdf,v,s=3, size = (6,10)):
    for a in range(len(acdf)):
        if np.sum([acdf[a:a+size[1]]>s*v])>size[0]:
            return a
    return len(acdf)-1


def optimal_sample(F, delta, eta, theta):
    gamma = 0.57721567

    M = (2*F/(eta/2-delta))
    M = M**2
    M = M *( np.log(np.log(1/(delta))) + np.log(1/theta))

    return M

def compute_d(tau,epsilon):
    d = np.sqrt(2)/(tau*epsilon)
    d = d * np.log(2*np.sqrt(2*np.pi)*(1+2/epsilon))
    d = int(d)
    return d

def toBinary(n):

    # Check if the number is Between 0 to 1 or Not
    if(n >= 1 or n <= 0):
        return "ERROR"

    answer = ""
    frac = 0.5
    answer = answer + "."

    # Setting a limit on length: 32 characters.
    while(n > 0):

        # Setting a limit on length: 32 characters
        if(len(answer) >= 64):
            return "ERROR"

        # Multiply n by 2 to check it 1 or 0
        b = n * 2
        if (b >= 1):

            answer = answer + "1"
            n = b - 1

        else:
            answer = answer + "0"
            n = b

    return answer

def toDecimal(binary):
    number =0
    for b in range(len(binary)):
        number += 2**(-b-1)*int(binary[b])

    return number


def truncate(ew,n):

    binary = toBinary(3*(ew + np.pi/3)/(2*np.pi))
    binary = binary[1:n+1]
#     print(binary)
    binary = binary[:-1]+'1'
    decimal = toDecimal(binary)
    value = 2*np.pi*decimal/3 - np.pi/3
#     print(binary,value)
    return value
