import os,runpy,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
runpy.run_path(os.path.join(os.path.dirname(__file__),'training_engineering.py'),run_name='__main__')
