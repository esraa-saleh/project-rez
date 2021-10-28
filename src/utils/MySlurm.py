import PyExpUtils.runner.Slurm as Slurm
import os
import json
from typing import Any, Dict, Iterator, Optional
import PyExpUtils.runner.parallel as Parallel

class MySlurmOptions(Slurm.Options):

    def __init__(self, d: Dict[str, Any]):
        super().__init__(d)
        self.num_gpus = d.get('num_gpus')
        self.gpu_type = d.get('gpu_type')
        self.cc_nodes = d.get('cc_nodes')
        self.gpu_enabled = False
        if ((self.num_gpus is not None) and (self.gpu_type is not None) and (self.num_gpus is not None)):
            self.gpu_enabled = True

    def gpu_specs_present(self):
        return self.gpu_enabled

    def cmdArgs(self):
        args = [
            f'--ntasks={self.tasks}',
            f'--mem-per-cpu={self.memPerCpu}',
            f'--output={self.output}',
        ]
        if(self.time != ''):
            args.insert(0, f'--time={self.time}')

        if(self.account != ''):
            args.insert(0, f'--account={self.account}')

        if(self.gpu_enabled):
            if(self.gpu_type != ''):
                args.extend([f'--nodes={self.cc_nodes}',
                             f'--gres=gpu:{self.gpu_type}:{self.num_gpus}'])
            else:
                args.extend([f'--nodes={self.cc_nodes}',
                             f'--gres=gpu:{self.num_gpus}'])


        if self.emailType is not None: args.append(f'--mail-type={self.emailType}')
        if self.email is not None: args.append(f'--main-user={self.email}')
        return ' '.join(args)


def fromFile(path: str):
    with open(path, 'r') as f:
        d = json.load(f)

    return MySlurmOptions(d)

# Based on Slurm.py in PyExpUtils
def buildParallel(executable: str, tasks: Iterator[Any], opts: Dict[str, Any] = {}, log_file: str = "runtask.log", with_gpu = True):
    # nodes = opts.get('nodes-per-process', 1)
    # threads = opts.get('threads-per-process', 1)

    build_dict = {
        'executable': f'--joblog logs/{log_file} {executable} {{}}',
        'tasks': tasks,
        'cores': opts['ntasks']
    }
    if (with_gpu):
        build_dict['executable'] = f'--joblog logs/{log_file} \'CUDA_VISIBLE_DEVICES=$(({{%}} - 1)) {executable} {{}}\''
    # if('gpus_per_process' in opts):
    #     num_gpus = opts['gpus_per_process']
    #     build_dict['executable']= f'--joblog logs/{log_file} srun -N{nodes} -n{threads} --gres=gpu:{num_gpus} --exclusive {executable}'

    return Parallel.build(build_dict)

# Based on Slurm.py in PyExpUtils
def schedule(script: str, opts: Optional[MySlurmOptions] = None, script_name: str = 'auto_slurm.sh', cleanup: bool = True):
    with open(script_name, 'w') as f:
        f.write(script)

    cmdArgs = ''
    if opts is not None:
        cmdArgs = opts.cmdArgs()

    os.system(f'sbatch {cmdArgs} {script_name}')

    if cleanup:
        os.remove(script_name)