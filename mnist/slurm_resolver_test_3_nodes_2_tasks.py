import tensorflow as tf
from cb_slurm_cluster_resolver import CbSlurmClusterResolver

if __name__ == '__main__':
    slurm_cluster_resolver = CbSlurmClusterResolver(
        jobs={
            'chief': 1,
            'worker': 4,
            'evaluator': 1,
        },
        port_base=8888)

    cluster_spec = slurm_cluster_resolver.cluster_spec()
    print(cluster_spec)
    print(slurm_cluster_resolver.get_task_info())
    print(slurm_cluster_resolver.master())
