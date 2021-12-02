import numpy as np
import femDataGen
import femRandInput

import time,os

def runRandRhoGen():
    femRandInput.set_seed(3)
    # 64 64 4096 -  458.1 MB
    # 64 64 512 -  57.2598 MB
    generator = femDataGen.femDataGenFilterCut(nx=64, ny=64) 
    generator.fillData(128, 128)
    stamp = femDataGen.caseStamp()
    generator.saveData('data/test2')

def runRandOptGen():
    femRandInput.set_seed(3)
    # 64 64 4096 -  458.1 MB
    # 64 64 512 -  57.2598 MB
    generator = femDataGen.femDataGenFilterCut(nx=64, ny=64,opterSeq=True) 
    generator.fillData(8, 16)
    stamp = femDataGen.caseStamp()
    generator.saveData('data/test_opt')


if __name__=='__main__':
    runRandOptGen()

    




    






