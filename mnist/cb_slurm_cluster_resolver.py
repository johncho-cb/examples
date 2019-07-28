from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import subprocess

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export


class CbSlurmClusterResolver(ClusterResolver):
  """Cerebras Cluster Resolver for Slurm workload manager.

  Based on slurm_cluster_resolver + deepsense-api/tensorflow_on_slurm.
  """

  def _pad_zeros(self, iterable, length):
    return (str(t).rjust(length, '0') for t in iterable)

  def _expand_ids(self, ids):
    ids = ids.split(',')
    result = []
    for id in ids:
        if '-' in id:
            begin, end = [int(token) for token in id.split('-')]
            result.extend(self._pad_zeros(range(begin, end+1), len(token)))
        else:
            result.append(id)
    return result

  def _resolve_hostnames(self):
    """Resolve host names of nodes allocated in current jobs.

    Returns:
      A list of node names as strings.
    """
    if 'SLURM_JOB_NODELIST' in os.environ:
      prefix, ids = re.findall("(.*)\[(.*)\]", os.environ["SLURM_JOB_NODELIST"])[0]
      ids = self._expand_ids(ids)
      hostlist = [prefix + str(id) for id in ids]
    else:
      raise RuntimeError('SLURM_JOB_NODELIST is not set')

    return hostlist

  def __init__(self,
               jobs,
               port_base=8888,
               rpc_layer='grpc'):
    """Creates a new CbSlurmClusterResolver object.

    Args:
      jobs: Dictionary with job names as key and number of tasks in the job as
        value
      port_base: The first port number to start with for processes on a node.
      rpc_layer: (Optional) The protocol TensorFlow uses to communicate between
        nodes. Defaults to 'grpc'.

    Returns:
      A CbClusterResolver object which can be used with distributed TensorFlow.
    """

    # check if launched by mpirun
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
      self._rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
      num_tasks = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    else:
      self._rank = int(os.environ['SLURM_PROCID'])
      num_tasks = int(os.environ['SLURM_NTASKS'])

    self._jobs = collections.OrderedDict(sorted(jobs.items()))
    self._port_base = port_base

    if 'SLURM_NTASKS_PER_NODE' in os.environ:
      self._tasks_per_node = int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
      self._tasks_per_node = 1

    self.task_type = None
    self.task_id = None
    self.rpc_layer = rpc_layer

    self._cluster_allocation = {}

    total_jobs = sum(self._jobs.values())
    if total_jobs != num_tasks:
      raise RuntimeError("Requested more tasks does not match assigned tasks.  jobs:{} != tasks:{}".
                         format(total_jobs, num_tasks))

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest instance group info.

    Returns:
      A ClusterSpec containing host information retrieved from Slurm's
        environment variables.
    """
    hostlist = self._resolve_hostnames()

    task_list = []
    self._cluster_allocation = {}

    for host in hostlist:
      for port_offset in range(self._tasks_per_node):

        host_addr = '%s:%d' % (host, self._port_base + port_offset)
        task_list.append(host_addr)

    cluster_rank_offset_start = 0
    cluster_rank_offset_end = 0

    for task_type, num_tasks in self._jobs.items():
      cluster_rank_offset_end = cluster_rank_offset_start + num_tasks

      self._cluster_allocation[task_type] = (
          task_list[cluster_rank_offset_start:cluster_rank_offset_end])

      if cluster_rank_offset_start <= self._rank < cluster_rank_offset_end:
        self.task_type = task_type
        self.task_id = self._rank - cluster_rank_offset_start

      cluster_rank_offset_start = cluster_rank_offset_end

    return ClusterSpec(self._cluster_allocation)

  def get_task_info(self):
    """Returns job name and task_id for the process which calls this.

    This returns the job name and task index for the process which calls this
    function according to its rank and cluster specification. The job name and
    task index are set after a cluster is constructed by cluster_spec otherwise
    defaults to None.

    Returns:
      A string specifying job name the process belongs to and an integner
        specifying the task index the process belongs to in that job.
    """
    return self.task_type, self.task_id

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master string for connecting to a TensorFlow master.

    Args:
      task_type: (Optional) Overrides the default auto-selected task type.
      task_id: (Optional) Overrides the default auto-slected task index.
      rpc_layer: (Optional) Overrides the default RPC protocol TensorFlow uses
        to communicate across nodes.

    Returns:
      A connection string for connecting to a TensorFlow master.
    """
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id

    if task_type is not None and task_id is not None:
      return format_master_url(
          self.cluster_spec().task_address(task_type, task_id),
          rpc_layer or self.rpc_layer)

    return ''
