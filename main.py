################## importing all the needed libraries ##################
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import itertools
import numpy as np
import bisect

####################### Question One #########################

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
