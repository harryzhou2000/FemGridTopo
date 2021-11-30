import numpy as np
import femDataGen
import femRandInput

import time,os



if __name__=='__main__':
    femRandInput.set_seed(3)
    # 64 64 4096 -  458.1 MB
    # 64 64 512 -  57.2598 MB
    generator = femDataGen.femDataGenFilterCut(nx=64, ny=64) 
    generator.fillData(128, 128)
    stamp = femDataGen.caseStamp()
    generator.saveData('data/test2')

    






