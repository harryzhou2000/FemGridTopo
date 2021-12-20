from re import T
import numpy as np
import femDataGen
import femRandInput
import argparse




import time,os

argP = argparse.ArgumentParser()
argP.add_argument('--seed', default=0, type=int)
argP.add_argument('--UseOpt', action='store_true', default=False)
argP.add_argument('--Static', action='store_true', default=False)
argP.add_argument('--nrho', type=int, default=16)
argP.add_argument('--nbnd', type=int, default=8)
argP.add_argument('--name', type=str, default='')

args = argP.parse_args()
seedname = '_S%04d_'%(args.seed)

def runRandRhoGen():
    femRandInput.set_seed(args.seed)
    # 64 64 4096 -  458.1 MB
    # 64 64 512 -  57.2598 MB
    generator = femDataGen.femDataGenFilterCut(nx=64, ny=64) 
    generator.fillData(args.nbnd, args.nrho)
    generator.saveData('data/test2', stamp=args.name+seedname+femDataGen.caseStamp())

def runRandOptGen():
    femRandInput.set_seed(args.seed)
    # 64 64 4096 -  458.1 MB
    # 64 64 512 -  57.2598 MB
    generator = femDataGen.femDataGenFilterCut(nx=64, ny=64,opterSeq=True) 
    generator.fillData(args.nbnd, args.nrho)
    generator.saveData('data/test_opt', stamp=args.name+seedname+femDataGen.caseStamp())

def runStaticGen():
    femRandInput.set_seed(args.seed)

    generator = femDataGen.femDataGenFilterCut(nx=64, ny=64, opterSeq=False, static=True)
    generator.fillData(args.nbnd, args.nrho)
    generator.saveData('data/test_static', stamp=args.name +
                       seedname+femDataGen.caseStamp())


if __name__=='__main__':
    # runRandOptGen()
    if args.UseOpt:
        runRandOptGen()
    elif args.Static:
        runStaticGen()
    else:
        runRandRhoGen()
    
    

    




    






