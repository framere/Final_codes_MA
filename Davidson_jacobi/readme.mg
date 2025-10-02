Block davidson: Just Tobias' code with the modified correction equation
Defl_algo: The defl1 algorithm used in previous cases, but the correction equation is modified, also we play a bit around with the buffer (see cheat sheet with all the slurms)
Mix_JD_dav: Here I try to implement an adaptive algorithm that switches between the JD and Davidson once I reached a specific threshold (based on Defl_algo)
Modified_defl: Eliminate 3 steps stability from defl 1 and use smaller buffer and betas
