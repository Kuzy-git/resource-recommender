import numpy as np

def recommend_cpu(cpu_mean, cpu_std, k=1.0):
    return cpu_mean + k * cpu_std

def recommend_memory(mem_max):
    return mem_max

def decision(current, recommended, delta=0.1):
    if recommended > current * (1 + delta):
        return "increase"
    elif recommended < current * (1 - delta):
        return "decrease"
    else:
        return "keep"