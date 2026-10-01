# Run after build-probe.py; test a conditional host-compiler argument.
import difflib
cm=repo/'cpp/0_Introduction/vectorAddDrv/CMakeLists.txt'
before=cm.read_text()
after=before.replace('add_custom_command(', 'set(CUDA_HOST_COMPILER_ARGS)\nif(CMAKE_CUDA_HOST_COMPILER)\n    list(APPEND CUDA_HOST_COMPILER_ARGS "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")\nendif()\n\nadd_custom_command(', 1).replace('-ccbin=${CMAKE_CUDA_HOST_COMPILER}', '${CUDA_HOST_COMPILER_ARGS}', 1)
# Replace only the original command argument, keeping the quoted list element.
after=before.replace('add_custom_command(', 'set(CUDA_HOST_COMPILER_ARGS)\nif(CMAKE_CUDA_HOST_COMPILER)\n    list(APPEND CUDA_HOST_COMPILER_ARGS "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")\nendif()\n\nadd_custom_command(', 1)
after=after.replace('-Wno-deprecated-gpu-targets -ccbin=${CMAKE_CUDA_HOST_COMPILER}', '-Wno-deprecated-gpu-targets ${CUDA_HOST_COMPILER_ARGS}')
cm.write_text(after)
patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/cpp/0_Introduction/vectorAddDrv/CMakeLists.txt',tofile='b/cpp/0_Introduction/vectorAddDrv/CMakeLists.txt'))
(root/'conditional-host-compiler.patch').write_text(patch)
logs=[]
for mode in ['default','explicit']:
 b=root/('fixed-'+mode)
 args=['cmake','-S',str(cm.parent),'-B',str(b),'-DCMAKE_CUDA_ARCHITECTURES=75','-DCMAKE_MODULE_PATH='+str(repo/'cmake')]
 if mode=='explicit': args+=['-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++']
 c=run(args)
 if c.returncode==0: run(['cmake','--build',str(b),'--target','generate_fatbin_vectorAdd','--verbose'])
run(['nvidia-smi'])
print(json.dumps({'logs':logs,'patch':patch}))
