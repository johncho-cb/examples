import tensorflow as tf
from cb_slurm_cluster_resolver import CbSlurmClusterResolver
import json

if __name__ == '__main__':
    slurm_cluster_resolver = CbSlurmClusterResolver(
        jobs={
            'chief': 1,
            'worker': 6,
            'evaluator': 0,
        },
        port_base=8888)

    cluster_spec = slurm_cluster_resolver.cluster_spec()
    print(json.dumps(cluster_spec.as_dict()))
    print(cluster_spec)
    print(slurm_cluster_resolver.get_task_info())
    print(slurm_cluster_resolver.master())
