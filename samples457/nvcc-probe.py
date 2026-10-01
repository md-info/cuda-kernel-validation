import pathlib, subprocess, json
p=pathlib.Path('/content/compiler-probe.cu')
p.write_text('__global__ void kernel() {}\n')
rows=[]
for flag in [[], ['-ccbin='], ['-ccbin=/usr/bin/g++']]:
 a=['nvcc','-arch=sm_75','-fatbin',str(p),'-o','/content/compiler-probe.fatbin']+flag
 r=subprocess.run(a,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 rows.append({'args':a,'exit':r.returncode,'output':r.stdout})
print(json.dumps(rows))
