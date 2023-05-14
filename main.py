################## importing all the needed libraries ##################
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import itertools
import numpy as np
import bisect

####################### Part One #########################

# >>>>>>>>>>>>>>>>>>>Daynmic approach<<<<<<<<<<<<<<<<<<:
def jobs_function(jobs):
    #sort job list in descinding order
    jobs = sorted(jobs, key=lambda x: x[2], reverse=True)
    
    #declaring variables
    n = len(jobs)
    maximum_profit = 0
    number_of_jobs = 0
    deadline = [False] * n
    
    #iterate and find slot for the job
    for job in jobs:
        for i in range(min(n, job[1])-1, -1, -1):
            if not deadline[i]:
                deadline[i] = True
                maximum_profit += job[2]
                number_of_jobs += 1
                break
    #return results
    return num_jobs, max_profit
  #apply input
jobs = [(1,4,20),(2,1,10),(3,1,40),(4,1,30)]
number_of_jobs, maximum_profit = jobs_function(jobs)
print("Number of jobs done:", number_of_jobs)
print("Maximum profit:", maximum_profit)

#>>>>>>>>>>> Naive  approach<<<<<<<<<<<:

#We use the itertools.permutations() function to generate all possible permutations of the jobs
import itertools
def get_max_pro(jobs):
    max_pro = 0
    n_done_jobs = 0
#For each permutation, we iterate through the jobs in order and calculate the total profit earned if we schedule them in that order
    for perm in itertools.permutations(jobs):
        pro = 0
        t = 0
        for job in perm:
            if t + 1 <= job[1]:
                t += 1
                pro += job[2]
#If the total profit earned is greater than the current maximum profit, we update the maximum profit and the number of jobs done
        if pro > max_pro:
            max_pro = pro
            n_done_jobs = len(perm)
    return (n_done_jobs, max_pro)

#Using the Example    
jobs = [(1,4,20),(2,1,10),(3,1,40),(4,1,30)]
print(get_max_pro(jobs))

# >>>>>>>>>>>>>>>>>>>>> Comparison between the two approaches<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def compare_functions():
    data = []
    
    for n in range(1, 11, 1):
        jobs = [(i, random.randint(1, 10), random.randint(1, 100)) for i in range(1, n+1)]
        
        # Measure execution time for the naive function
        start_time = time.time()
        get_max_pro(jobs)
        naive_time = time.time() - start_time
        
        # Measure execution time for the dynamic programming function
        start_time = time.time()
        jobs_function(jobs)
        dynamic_time = time.time() - start_time
        
        data.append({'N': n, 'Naive Time': naive_time, 'Dynamic Time': dynamic_time})
    
    df = pd.DataFrame(data)
    return df

df = compare_functions()
print(df)
plt.plot(df['N'], df['Naive Time'], label='Naive')
plt.plot(df['N'], df['Dynamic Time'], label='Dynamic')
plt.xlabel('Input Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Comparison of Execution Times')
plt.legend()
plt.show()

####################### Part Two #########################

# function to get the max profit by try all the possible combination without any overlap 
def brute_force( jobs, S, F, P, n):
    profits = []
    for n in range(1,n):
        Combinations = list(combinations(jobs,n))
        for c in Combinations:
            overlap = False
            profit = 0
            for i in range(len(c)):
                job_i = c[i]
                profit += P[job_i]
                for j in range(i+1,len(c)):
                    job_j = c[j]
                    if F[job_i] > S[job_j] and F[job_j] > S[job_i]:
                        overlap = True
                if overlap:
                    break
            if not(overlap):
                profits.append(profit)
                
    return max(profits)

# let's define another approach to slove the problem
def greedy_recursive_sol( S, F, spliter, n):
    """
    that algorithm assumes the data are sorted by finish time
    ,the ids start from 0 to n and there exists a redundent job k ; S[k] = F[k] = 0.
    inputs :
       S       : list ------> start time of the jobs 
       F       : list------> finish time of the jobs
       spliter : 0 -------> subproblem spliter such that A.spliter  is the activity 
       n       : integer-------> N.O jobs
    output :
       set represent the greedy best solu 'optimal'
    
    """
    m = spliter + 1    
    while (m <= n) & (S[m] < F[spliter]) :
        m = m + 1
        
    if m < n :
        return {m}.union(greedy_recursive_sol( S, F, m, n))
    else :
        return {m}
    
def dynamic_solu( S, F, P):
    """
    that algorithm assumes the data are sorted by finish time 
    inputs :
       S : list ------> start time of the jobs
       F : list------> finish time of the jobs
       P : list------> profit of each job
    output :
       the maximum profit
    """
    # if the all the jobs have the same profit then use a greedy approach O(n)
    if (len(set(P)) == 1):
        jobs = greedy_recursive_sol(S, F, 0, len(S)-1)
        return len(jobs)*P[0]
    else :
        # if not use the dynamic approach  O( n*lg(n) )
        memoize = [[0,0]]
        for s, f, p in zip(S,F,P):
            i = bisect.bisect(memoize, [s + 1]) - 1
            if (memoize[i][1] + p) > memoize[-1][1]:
                memoize.append([f, memoize[i][1] + p])        
        return memoize[-1][1]   



