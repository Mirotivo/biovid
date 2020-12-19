# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr


 pip install sampen

"""
#       Sample Entropy
from base_preprocessing import base_preprocessing
import math
import numpy as np
class mav_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        def samp_entropy(X, M, R):
            def embed_seq(X,Tau,D):
                	N =len(X)
                	if D * Tau > N:
                		print("Cannot build such a matrix, because D * Tau > N") 
                		exit()
                	if Tau<1:
                		print("Tau has to be at least 1")
                		exit()
                	Y=np.zeros((N - (D - 1) * Tau, D))
                	for i in range(0, N - (D - 1) * Tau):
                		for j in range(0, D):
                			Y[i][j] = X[i + j * Tau]
                	return Y
        
            def in_range(Template, Scroll, Distance):
                	for i in range(0,  len(Template)):
                			if abs(Template[i] - Scroll[i]) > Distance:
                			     return False
                	return True
        
            
            """Computer sample entropy (SampEn) of series X, specified by M and R.
            
            	SampEn is very close to ApEn. 
            
            	Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
            	embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of Em 
            	is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension are
            	1 and M-1 respectively. Such a matrix can be built by calling pyeeg function 
            	as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only 
            	difference with Em is that the length of each embedding sequence is M + 1
            
            	Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elments 
            	are	Em[i][k] and Em[j][k] respectively. The distance between Em[i] and Em[j]
            	is defined as 1) the maximum difference of their corresponding scalar 
            	components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two 1-D
            	vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance between them 
            	is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the value of R is
            	defined as 20% - 30% of standard deviation of X. 
            
            	Pick Em[i] as a template, for all j such that 0 < j < N - M , we can 
            	check whether Em[j] matches with Em[i]. Denote the number of Em[j],  
            	which is in the range of Em[i], as k[i], which is the i-th element of the 
            	vector k.
            
            	We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.
            
            	The SampEn is defined as log(sum(Cm)/sum(Cmp))
            
            	References
            	----------
            
            	Costa M, Goldberger AL, Peng C-K, Multiscale entropy analysis of biolgical
            	signals, Physical Review E, 71:021906, 2005
            
            	See also
            	--------
            	ap_entropy: approximate entropy of a time series
            
            
            	Notes
            	-----
            	Extremely slow computation. Do NOT use if your dataset is not small and you
            	are not patient enough.
            
            	"""
        
            N = len(X)
            Em = embed_seq(X, 1, M)	
            Emp = embed_seq(X, 1, M + 1)
            Cm, Cmp = np.zeros(N - M - 1) + 1e-100, np.zeros(N - M - 1) + 1e-100
            # in case there is 0 after counting. Log(0) is undefined.
        
            for i in range(0, N - M):
                for j in range(i + 1, N - M): # no self-match
                #			if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1 
                    if in_range(Em[i], Em[j], R):
                        Cm[i] += 1
                #			if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
                        if abs(Emp[i][-1] - Emp[j][-1]) <= R: # check last one
                            Cmp[i] += 1
            Samp_En = math.log(sum(Cm)/sum(Cmp))
            return Samp_En
        return samp_entropy(list_signal,2,3)
