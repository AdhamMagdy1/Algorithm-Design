# Algorithm Design Project
This project is part of our college course on Algorithm Design. We have been given a problem and we need to design an algorithm to solve it using Python.
## Problem Description
This project involves solving two problems related to job scheduling:
### Part A
Given a set of N jobs where each job has a deadline and profit associated with it. Each job takes 1 unit of time to complete and only one job can be scheduled at a time. We earn the profit associated with a job if and only if the job is completed by its deadline. The task is to find the maximum profit that can be earned and the number of jobs that can be completed within their respective deadlines. The jobs will be given in the form (Jobid, Deadline, Profit) associated with that Job.

Example 1:

 Input:  
` Jobs = {(1, 4, 20), (2, 1, 10), (3, 1, 40), (4, 1, 30)} `  
 Output:  
` Number of jobs done: 2
Maximum Profit: 60 `  

### Part B
You are given a list of N jobs with their start time, end time, and the profit you can earn by doing that job. Your task is to find the maximum profit you can earn by picking up some (or all) jobs, ensuring no two jobs overlap. If you choose a job that ends at time X, you will be able to start another job that starts at time X. The jobs will be given in the form {Jobid, StartTime, EndTime, Profit}.

Example 1:

 Input:  
` Jobs = {(1, 6, 6), (2, 5, 5), (5, 7, 5), (6, 8, 3)} `  
 Output:  
` Maximum Profit: 11 `  

## Approach
We will be using dynamic programming to solve each problem. Dynamic programming is a technique for solving complex problems by breaking them down into smaller subproblems and storing the results of each subproblem to avoid redundant calculations.

In addition, we will also implement the naive method for each problem and compare the results and efficiency of both methods. The naive method involves solving the problem directly without taking advantage of any substructure or repeated calculations. By comparing the results and efficiency of both methods, we can gain insights into the tradeoffs between accuracy and efficiency and make informed decisions about which approach to use in different scenarios.

We believe that this approach will provide a comprehensive and effective way to solve the problems  and enable us to gain a deeper understanding of dynamic programming and its applications.
## Instructions
To run the program, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the root directory of the repository.
3. Run the program using the following command: `python main.py`.
4. Follow the prompts to input any necessary information.

# Contributors
- Adham Magdy          (https://github.com/AdhamMagdy1)
- Eslam Tarek          (https://github.com/EslamTarekFarouk)
- Ahmed Ashraf         (https://github.com/Ahmedashraf547)
- Ahmed Shehata        (https://github.com/Ahmedashraf547)
- Elhussien Ahmed      (https://github.com/hussienahmed859)

