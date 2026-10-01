import pathlib, subprocess, json, os
root=pathlib.Path('/content/samples457')
root.mkdir(exist_ok=True)
logs=[]
def run(args):
 r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 logs.append({'args':args,'exit':r.returncode,'output':r.stdout})
 return r
repo=root/'repo'
if not repo.exists(): run(['git','clone','--quiet','https://github.com/NVIDIA/cuda-samples.git',str(repo)])
run(['git','-C',str(repo),'fetch','--quiet','origin','pull/457/head'])
for label,rev in [('base','5443602d89ed99aede2e4b7bf329daddeadb320e'),('head','70c867aaf00bcca8729c6a422c00c2d07ffb8500')]:
 run(['git','-C',str(repo),'checkout','--quiet',rev])
 for mode in ['default','explicit']:
  b=root/(label+'-'+mode)
  args=['cmake','-S',str(repo/'cpp/0_Introduction/vectorAddDrv'),'-B',str(b),'-DCMAKE_CUDA_ARCHITECTURES=75','-DCMAKE_MODULE_PATH='+str(repo/'cmake')]
  if mode=='explicit': args+=['-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++']
  c=run(args)
  if c.returncode==0: run(['cmake','--build',str(b),'--target','generate_fatbin_vectorAdd','--verbose'])
run(['nvcc','--version'])
run(['cmake','--version'])
run(['g++','--version'])
(root/'build-results.json').write_text(json.dumps(logs,indent=2))
print(json.dumps(logs))
