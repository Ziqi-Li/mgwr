import numpy as np
import subprocess
import os
import mgwr
from tempfile import NamedTemporaryFile
from spglm.utils import cache_readonly

class FastGWR(object):
    def __init__(self, coords, y, X, fixed=False, constant=True, bw=None):
        self.y = y
        self.n = y.shape[0]
        if constant:
            self.X = np.hstack([np.ones((self.n,1)),X])
        else:
            self.X = X
        self.k = self.X.shape[1]
        self.coords = coords
        self.bw = bw
        self.fixed = fixed

    def fit(self,nproc):
        data = np.hstack([np.array(self.coords),self.y,self.X])
        datafile = NamedTemporaryFile(delete=False)
        np.savetxt(datafile, data, delimiter=',',comments='')
        datafile.close()
        resultfile = NamedTemporaryFile(delete=False)
        resultfile.name
        
        mpi_cmd = 'mpiexec' + ' -n ' + str(nproc) + ' python ' + os.path.abspath(os.path.join(mgwr.__file__, os.pardir)) + '/FastGWR_mpi.py ' + '-data ' + datafile.name + ' -out ' + resultfile.name
        
        if self.bw:
            mpi_cmd += ' -bw ' + str(self.bw)
        if self.fixed:
            mpi_cmd += ' -f 1'
        #print(mpi_cmd)
        subprocess.run(mpi_cmd, shell=True)
        
        return FastGWRResults(self,rslt=resultfile.name)


class FastGWRResults(object):
    def __init__(self, model, rslt):
        output = np.genfromtxt(rslt, dtype=float, delimiter=',',skip_header=False)
        self.k = model.k
        self.n = model.n
        self.y = model.y
        self.index = output[:,0]
        self.influ = output[:,2]
        self.resid_response = output[:,1]
        self.params = output[:,3:(3+self.k)]
        self.CCT = output[:,-self.k:]
    
    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)
    
    @cache_readonly
    def bse(self):
        return np.sqrt(self.CCT*self.sigma2)
    
    @cache_readonly
    def sigma2(self):
        return (self.resid_ss / (self.n-self.tr_S))
    
    @cache_readonly
    def aicc(self):
        RSS = self.resid_ss
        trS = self.tr_S
        n = self.n
        aicc = n*np.log(RSS/n) + n*np.log(2*np.pi) + n*(n+trS)/(n-trS-2.0)
        return aicc
    
    @cache_readonly
    def resid_ss(self):
        return np.sum(self.resid_response**2)
    
    @cache_readonly
    def predy(self):
        return self.y.reshape(-1) - self.resid_response



