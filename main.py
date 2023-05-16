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
    #sort job list in descinding order according to their profits
    jobs = sorted(jobs, key=lambda x: x[2], reverse=True)
    
    #declaring variables
    n = len(jobs)
    maximum_profit = 0
    number_of_jobs = 0
    deadline = [False] * n
    
    #iterates over each job and tries to find a slot for it such that its deadline is not missed
    for job in jobs:
        for i in range(min(n, job[1])-1, -1, -1):
            #If a slot is found the job is assigned to that slot and the maximum profit and the number of jobs assigned are updated 
            #If no slot is found the job is skipped
            if not deadline[i]:
                deadline[i] = True
                maximum_profit += job[2]
                number_of_jobs += 1
                break
    #return results
    return number_of_jobs, maximum_profit
#apply input
jobs = [(1,4,20),(2,1,10),(3,1,40),(4,1,30)]
number_of_jobs, maximum_profit = jobs_function(jobs)
print("Number of jobs done:", number_of_jobs)
print("Maximum profit:", maximum_profit)

#>>>>>>>>>>> Naive  approach<<<<<<<<<<<:

# Define the function to find the maximum profit
def max_pro(jobs):
    # Get the number of jobs
    n = len(jobs)
    # Initialize the maximum profit and the set of jobs that yield the maximum profit
    max_pro = 0
    max_jobs = []
    # Loop through all possible subsets of jobs
    for i in range(2**n):
        # Initialize the set of selected jobs and the current profit
        sel_jobs = []
        curr_pro = 0
        # Loop through all jobs
        for j in range(n):
            # If the jth bit of i is 1, add job j to the set of selected jobs
            if (i >> j) & 1:
                sel_jobs.append(jobs[j])
        # Sort the selected jobs by their deadlines
        sel_jobs = sorted(sel_jobs, key=lambda x: x[1])
        # Initialize the current time
        curr_time = 0
        # Loop through the selected jobs
        for job in sel_jobs:
            # If the job can be completed within its deadline, update the current time and profit
            if curr_time + 1 <= job[1]:
                curr_time += 1
                curr_pro += job[2]
        # If the current set of jobs yields a higher profit than the previous maximum, update the maximum profit and the set of jobs
        if curr_pro > max_pro:
            max_pro = curr_pro
            max_jobs = sel_jobs
    # Convert the set of jobs that yield the maximum profit to a list of job IDs and return the length of the list and the maximum profit as a tuple
    max_jobs = [job[0] for job in max_jobs]
    return len(max_jobs), max_pro

# Define the set of jobs
jobs = [(1,4,20),(2,1,10),(3,1,40),(4,1,30)]
# Find the maximum profit and the set of jobs that yield the maximum profit
result = max_pro(jobs)
# Print the result
print(result)

# >>>>>>>>>>>>>>>>>>>>> Comparison between the two approaches<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def compare_functions():
    data = []
    
    for n in range(1, 11, 1):
        jobs = [(i, random.randint(1, 10), random.randint(1, 100)) for i in range(1, n+1)]
        
        # Measure execution time for the naive function
        start_time = time.time()
        max_pro(jobs)
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



