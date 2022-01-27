---
title : "[Pytorch] DataLoader parameter별 용도"
category :
  - ML
tag :
  - pytorch
  - dataloader
  - parameter
  - sampler
  - num_workers
  - pin_memory
  - collate_fn
sidebar_main : true
author_profile : true
use_math : true
header:
  overlay_image : https://pytorch.org/tutorials/_static/img/thumbnails/cropped/Introduction-to-TorchScript.png
  overlay_filter: 0.5
published : true
---
pytorch reference 문서를 다 외우면 얼마나 편할까!!

PyTorch는 `torch.utils.data.Dataset`으로 Custom Dataset을 만들고, `torch.utils.data.DataLoader`로 데이터를 불러옵니다.

하지만 하다보면 데이터셋에 어떤 설정을 주고 싶고, 이를 조정하는 파라미터가 꽤 있다는 걸 알 수 있습니다.
그래서 이번에는 torch의 `DataLoader`의 몇 가지 기능을 살펴보겠습니다.

## Overview

![pytorch_dataloader](/_resources/pytorch_dataloader.png)

## DataLoader Parameters

### dataset

- *`Dataset`*

`torch.utils.data.Dataset`의 객체를 사용해야 합니다.

참고로 torch의 `dataset`은 2가지 스타일이 있습니다.

- **Map-style dataset**
  - index가 존재하여 data[index]로 데이터를 참조할 수 있음
  - For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk.
  - `__getitem__`과 `__len__` 선언 필요
- **Iterable-style dataset**
  - random으로 읽기에 어렵거나, data에 따라 batch size가 달라지는 데이터(dynamic batch size)에 적합
  - 비교하자면 stream data, real-time log 등에 적합
  - `__iter__` 선언 필요

이 점을 유의하며 아래의 파라미터 설명을 읽으면 더 이해가 쉽습니다.

### batch_size

- *`int`, optional, default=`1`*

**배치(batch)**의 크기입니다. 데이터셋에 50개의 데이터가 있고, batch_size가 10라면 총 50/10=5, 즉 5번의 iteration만 지나면 모든 데이터를 볼 수 있습니다.

이 경우 반복문을 돌리면 `(batch_size, *(data.shape))`의 형태의 `Tensor`로 데이터가 반환됩니다. dataset에서 return하는 모든 데이터는 Tensor로 변환되어 오니 Tensor로 변환이 안되는 데이터는 에러가 납니다.

### shuffle

- *`bool`, optional, default=`False`*

데이터를 DataLoader에서 섞어서 사용하겠는지를 설정할 수 있습니다.
실험 재현을 위해 `torch.manual_seed`를 고정하는 것도 포인트입니다.

> 그냥 Dataset에서 initialize할 때, random.shuffle로 섞을 수도 있습니다.

### sampler

- *`Sampler`, optional*

`torch.utils.data.Sampler` 객체를 사용합니다.

dataset은 inex로 data를 가져오도록 설계되었기 때문에, shuffle을 하기 위해서 index를 적절히 섞어 주면 된다. 그 것을 구현한 것이 `Sampler`이다.
- 매 step 마다 다음 index를 yield하면 됨.
  - `__len__`과 `__iter__`를 구현하면 된다.

```python
point_sampler = RandomSampler(map_dataset)
dataloader = torch.utils.data.DataLoader(map_dataset,
                                         batch_size=4,
                                         sampler=point_sampler)
for data in dataloader:
    print(data['input'].shape, data['label'])
```

sampler는 index를 컨트롤하는 방법입니다. 데이터의 index를 원하는 방식대로 조정합니다.
즉 index를 컨트롤하기 때문에 설정하고 싶다면 `shuffle` 파라미터는 `False`(기본값)여야 합니다.

map-style에서 컨트롤하기 위해 사용하며 `__len__`과 `__iter__`를 구현하면 됩니다.
그 외의 미리 선언된 Sampler는 다음과 같습니다.

- `SequentialSampler` : 항상 같은 순서
- `RandomSampler` : 랜덤, replacemetn 여부 선택 가능, 개수 선택 가능
- `SubsetRandomSampler` : 랜덤 리스트, 위와 두 조건 불가능
- `WeigthRandomSampler` : 가중치에 따른 확률
- `BatchSampler` : batch단위로 sampling 가능
- `DistributedSampler` : 분산처리 (`torch.nn.parallel.DistributedDataParallel`과 함께 사용)

### batch_sampler
batch 단위로 sampling할 때 쓴다.
- 매 step마다 `index의 list`를 반환하면 batch_sampler로 쓸 수 있음

- *`Sampler`, optional*

위와 거의 동일하므로 생략합니다.

### num_workers

- *`int`, optional, default=`0`*

데이터 로딩에 사용하는 subprocess개수입니다. (멀티프로세싱)

기본값이 0인데 이는 data가 main process로 불러오는 것을 의미합니다.
그럼 많이 사용하면 좋지 않은가? 라고 질문하실 수도 있습니다.

하지만 데이터를 불러 CPU와 GPU 사이에서 많은 교류가 일어나면 오히려 병목이 생길 수 있습니다.
이것도 trade-off관계인데, 이와 관련하여는 다음 글을 추천합니다.

- [DataLoader num_workers에 대한 고찰](https://jybaek.tistory.com/799)

### collate_fn
batch_sampler로 묶이 이후에는, collate_fn을 호출해서 batch로 묶는다.
- `collate_fn([dataset[i] for i in indices])`

dataset이 variable length면 바로 못 묶이고 에러가 나므로, `collate_fn`을 만들어서 넘겨줘야 함.
- *callable, optional*

map-style 데이터셋에서 sample list를 batch 단위로 바꾸기 위해 필요한 기능입니다.
zero-padding이나 Variable Size 데이터 등 데이터 사이즈를 맞추기 위해 많이 사용합니다.

### pin_memory

- *`bool`, optional*

`True`러 선언하면, 데이터로더는 Tensor를 CUDA 고정 메모리에 올립니다.

어떤 상황에서 더 빨라질지는 다음 글을 참고합시다.

- discuss.Pytorch : [When to set pin_memory to true?](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723)

### drop_last

- *`bool`, optional*

`batch` 단위로 데이터를 불러온다면, batch_size에 따라 마지막 batch의 길이가 달라질 수 있습니다.
예를 들어 data의 개수는 27개인데, batch_size가 5라면 마지막 batch의 크기는 2가 되겠죠.

batch의 길이가 다른 경우에 따라 loss를 구하기 귀찮은 경우가 생기고, batch의 크기에 따른 의존도 높은 함수를 사용할 때 걱정이 되는 경우 마지막 batch를 사용하지 않을 수 있습니다.

### time_out

- *numeric, optional, default=`0`*

양수로 주어지는 경우, DataLoader가 data를 불러오는데 제한시간입니다.

### worker_init_fn

- *callable, optional, default='None'*

num_worker가 개수라면, 이 파라미터는 어떤 worker를 불러올 것인가를 리스트로 전달합니다.

> 아래 2개는 언제 사용하는걸까요?

## Reference

- official : [torch.utils.data](https://pytorch.org/docs/stable/data.html)

- Hulk의 개인 공부용 블로그 : [pytorch dataset 정리](https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/) : 핵심적인 함수의 사용법들과 커스텀 클래스 선언이 궁금하신 분들에게 추천합니다.

---
torch.utils.data
===================================

.. automodule:: torch.utils.data

At the heart of PyTorch data loading utility is the :class:`torch.utils.data.DataLoader`
class.  It represents a Python iterable over a dataset, with support for

* `map-style and iterable-style datasets <Dataset Types_>`_,

* `customizing data loading order <Data Loading Order and Sampler_>`_,

* `automatic batching <Loading Batched and Non-Batched Data_>`_,

* `single- and multi-process data loading <Single- and Multi-process Data Loading_>`_,

* `automatic memory pinning <Memory Pinning_>`_.

These options are configured by the constructor arguments of a
:class:`~torch.utils.data.DataLoader`, which has signature::

    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, *, prefetch_factor=2,
               persistent_workers=False)

The sections below describe in details the effects and usages of these options.

Dataset Types
-------------

The most important argument of :class:`~torch.utils.data.DataLoader`
constructor is :attr:`dataset`, which indicates a dataset object to load data
from. PyTorch supports two different types of datasets:

* `map-style datasets <Map-style datasets_>`_,

* `iterable-style datasets <Iterable-style datasets_>`_.

#### Map-style datasets

A map-style dataset is one that implements the :meth:`__getitem__` and
:meth:`__len__` protocols, and represents a map from (possibly non-integral)
indices/keys to data samples.

For example, such a dataset, when accessed with ``dataset[idx]``, could read
the ``idx``-th image and its corresponding label from a folder on the disk.

See :class:`~torch.utils.data.Dataset` for more details.

#### Iterable-style datasets


An iterable-style dataset is an instance of a subclass of :class:`~torch.utils.data.IterableDataset`
that implements the :meth:`__iter__` protocol, and represents an iterable over
data samples. This type of datasets is particularly suitable for cases where
random reads are expensive or even improbable, and where the batch size depends
on the fetched data.

For example, such a dataset, when called ``iter(dataset)``, could return a
stream of data reading from a database, a remote server, or even logs generated
in real time.

See :class:`~torch.utils.data.IterableDataset` for more details.

.. note:: When using an :class:`~torch.utils.data.IterableDataset` with
          `multi-process data loading <Multi-process data loading_>`_. The same
          dataset object is replicated on each worker process, and thus the
          replicas must be configured differently to avoid duplicated data. See
          :class:`~torch.utils.data.IterableDataset` documentations for how to
          achieve this.

Data Loading Order and :class:`~torch.utils.data.Sampler`
---------------------------------------------------------

For `iterable-style datasets <Iterable-style datasets_>`_, data loading order
is entirely controlled by the user-defined iterable. This allows easier
implementations of chunk-reading and dynamic batch size (e.g., by yielding a
batched sample at each time).

The rest of this section concerns the case with
`map-style datasets <Map-style datasets_>`_. :class:`torch.utils.data.Sampler`
classes are used to specify the sequence of indices/keys used in data loading.
They represent iterable objects over the indices to datasets.  E.g., in the
common case with stochastic gradient decent (SGD), a
:class:`~torch.utils.data.Sampler` could randomly permute a list of indices
and yield each one at a time, or yield a small number of them for mini-batch
SGD.

A sequential or shuffled sampler will be automatically constructed based on the :attr:`shuffle` argument to a :class:`~torch.utils.data.DataLoader`.
Alternatively, users may use the :attr:`sampler` argument to specify a
custom :class:`~torch.utils.data.Sampler` object that at each time yields
the next index/key to fetch.

A custom :class:`~torch.utils.data.Sampler` that yields a list of batch
indices at a time can be passed as the :attr:`batch_sampler` argument.
Automatic batching can also be enabled via :attr:`batch_size` and
:attr:`drop_last` arguments. See
`the next section <Loading Batched and Non-Batched Data_>`_ for more details
on this.

.. note::
  Neither :attr:`sampler` nor :attr:`batch_sampler` is compatible with
  iterable-style datasets, since such datasets have no notion of a key or an
  index.

Loading Batched and Non-Batched Data
------------------------------------

:class:`~torch.utils.data.DataLoader` supports automatically collating
individual fetched data samples into batches via arguments
:attr:`batch_size`, :attr:`drop_last`, and :attr:`batch_sampler`.


Automatic batching (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most common case, and corresponds to fetching a minibatch of
data and collating them into batched samples, i.e., containing Tensors with
one dimension being the batch dimension (usually the first).

When :attr:`batch_size` (default ``1``) is not ``None``, the data loader yields
batched samples instead of individual samples. :attr:`batch_size` and
:attr:`drop_last` arguments are used to specify how the data loader obtains
batches of dataset keys. For map-style datasets, users can alternatively
specify :attr:`batch_sampler`, which yields a list of keys at a time.

.. note::
  The :attr:`batch_size` and :attr:`drop_last` arguments essentially are used
  to construct a :attr:`batch_sampler` from :attr:`sampler`. For map-style
  datasets, the :attr:`sampler` is either provided by user or constructed
  based on the :attr:`shuffle` argument. For iterable-style datasets, the
  :attr:`sampler` is a dummy infinite one. See
  `this section <Data Loading Order and Sampler_>`_ on more details on
  samplers.

.. note::
  When fetching from
  `iterable-style datasets <Iterable-style datasets_>`_ with
  `multi-processing <Multi-process data loading_>`_, the :attr:`drop_last`
  argument drops the last non-full batch of each worker's dataset replica.

After fetching a list of samples using the indices from sampler, the function
passed as the :attr:`collate_fn` argument is used to collate lists of samples
into batches.

In this case, loading from a map-style dataset is roughly equivalent with::

    for indices in batch_sampler:
        yield collate_fn([dataset[i] for i in indices])

and loading from an iterable-style dataset is roughly equivalent with::

    dataset_iter = iter(dataset)
    for indices in batch_sampler:
        yield collate_fn([next(dataset_iter) for _ in indices])

A custom :attr:`collate_fn` can be used to customize collation, e.g., padding
sequential data to max length of a batch. See
`this section <dataloader-collate_fn_>`_ on more about :attr:`collate_fn`.

Disable automatic batching
^^^^^^^^^^^^^^^^^^^^^^^^^^

In certain cases, users may want to handle batching manually in dataset code,
or simply load individual samples. For example, it could be cheaper to directly
load batched data (e.g., bulk reads from a database or reading continuous
chunks of memory), or the batch size is data dependent, or the program is
designed to work on individual samples.  Under these scenarios, it's likely
better to not use automatic batching (where :attr:`collate_fn` is used to
collate the samples), but let the data loader directly return each member of
the :attr:`dataset` object.

When both :attr:`batch_size` and :attr:`batch_sampler` are ``None`` (default
value for :attr:`batch_sampler` is already ``None``), automatic batching is
disabled. Each sample obtained from the :attr:`dataset` is processed with the
function passed as the :attr:`collate_fn` argument.

**When automatic batching is disabled**, the default :attr:`collate_fn` simply
converts NumPy arrays into PyTorch Tensors, and keeps everything else untouched.

In this case, loading from a map-style dataset is roughly equivalent with::

    for index in sampler:
        yield collate_fn(dataset[index])

and loading from an iterable-style dataset is roughly equivalent with::

    for data in iter(dataset):
        yield collate_fn(data)

See `this section <dataloader-collate_fn_>`_ on more about :attr:`collate_fn`.

.. _dataloader-collate_fn:

Working with :attr:`collate_fn`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of :attr:`collate_fn` is slightly different when automatic batching is
enabled or disabled.

**When automatic batching is disabled**, :attr:`collate_fn` is called with
each individual data sample, and the output is yielded from the data loader
iterator. In this case, the default :attr:`collate_fn` simply converts NumPy
arrays in PyTorch tensors.

**When automatic batching is enabled**, :attr:`collate_fn` is called with a list
of data samples at each time. It is expected to collate the input samples into
a batch for yielding from the data loader iterator. The rest of this section
describes behavior of the default :attr:`collate_fn` in this case.

For instance, if each data sample consists of a 3-channel image and an integral
class label, i.e., each element of the dataset returns a tuple
``(image, class_index)``, the default :attr:`collate_fn` collates a list of
such tuples into a single tuple of a batched image tensor and a batched class
label Tensor. In particular, the default :attr:`collate_fn` has the following
properties:

* It always prepends a new dimension as the batch dimension.

* It automatically converts NumPy arrays and Python numerical values into
  PyTorch Tensors.

* It preserves the data structure, e.g., if each sample is a dictionary, it
  outputs a dictionary with the same set of keys but batched Tensors as values
  (or lists if the values can not be converted into Tensors). Same
  for ``list`` s, ``tuple`` s, ``namedtuple`` s, etc.

Users may use customized :attr:`collate_fn` to achieve custom batching, e.g.,
collating along a dimension other than the first, padding sequences of
various lengths, or adding support for custom data types.

Single- and Multi-process Data Loading
--------------------------------------

A :class:`~torch.utils.data.DataLoader` uses single-process data loading by
default.

Within a Python process, the
`Global Interpreter Lock (GIL) <https://wiki.python.org/moin/GlobalInterpreterLock>`_
prevents true fully parallelizing Python code across threads. To avoid blocking
computation code with data loading, PyTorch provides an easy switch to perform
multi-process data loading by simply setting the argument :attr:`num_workers`
to a positive integer.

Single-process data loading (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this mode, data fetching is done in the same process a
:class:`~torch.utils.data.DataLoader` is initialized.  Therefore, data loading
may block computing.  However, this mode may be preferred when resource(s) used
for sharing data among processes (e.g., shared memory, file descriptors) is
limited, or when the entire dataset is small and can be loaded entirely in
memory.  Additionally, single-process loading often shows more readable error
traces and thus is useful for debugging.


Multi-process data loading
^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting the argument :attr:`num_workers` as a positive integer will
turn on multi-process data loading with the specified number of loader worker
processes.

In this mode, each time an iterator of a :class:`~torch.utils.data.DataLoader`
is created (e.g., when you call ``enumerate(dataloader)``), :attr:`num_workers`
worker processes are created. At this point, the :attr:`dataset`,
:attr:`collate_fn`, and :attr:`worker_init_fn` are passed to each
worker, where they are used to initialize, and fetch data. This means that
dataset access together with its  internal IO, transforms
(including :attr:`collate_fn`) runs in the worker process.

:func:`torch.utils.data.get_worker_info()` returns various useful information
in a worker process (including the worker id, dataset replica, initial seed,
etc.), and returns ``None`` in main process. Users may use this function in
dataset code and/or :attr:`worker_init_fn` to individually configure each
dataset replica, and to determine whether the code is running in a worker
process. For example, this can be particularly helpful in sharding the dataset.

For map-style datasets, the main process generates the indices using
:attr:`sampler` and sends them to the workers. So any shuffle randomization is
done in the main process which guides loading by assigning indices to load.

For iterable-style datasets, since each worker process gets a replica of the
:attr:`dataset` object, naive multi-process loading will often result in
duplicated data. Using :func:`torch.utils.data.get_worker_info()` and/or
:attr:`worker_init_fn`, users may configure each replica independently. (See
:class:`~torch.utils.data.IterableDataset` documentations for how to achieve
this. ) For similar reasons, in multi-process loading, the :attr:`drop_last`
argument drops the last non-full batch of each worker's iterable-style dataset
replica.

Workers are shut down once the end of the iteration is reached, or when the
iterator becomes garbage collected.

.. warning::
  It is generally not recommended to return CUDA tensors in multi-process
  loading because of many subtleties in using CUDA and sharing CUDA tensors in
  multiprocessing (see :ref:`multiprocessing-cuda-note`). Instead, we recommend
  using `automatic memory pinning <Memory Pinning_>`_ (i.e., setting
  :attr:`pin_memory=True`), which enables fast data transfer to CUDA-enabled
  GPUs.

Platform-specific behaviors
"""""""""""""""""""""""""""

Since workers rely on Python :py:mod:`multiprocessing`, worker launch behavior is
different on Windows compared to Unix.

* On Unix, :func:`fork()` is the default :py:mod:`multiprocessing` start method.
  Using :func:`fork`, child workers typically can access the :attr:`dataset` and
  Python argument functions directly through the cloned address space.

* On Windows, :func:`spawn()` is the default :py:mod:`multiprocessing` start method.
  Using :func:`spawn()`, another interpreter is launched which runs your main script,
  followed by the internal worker function that receives the :attr:`dataset`,
  :attr:`collate_fn` and other arguments through :py:mod:`pickle` serialization.

This separate serialization means that you should take two steps to ensure you
are compatible with Windows while using multi-process data loading:

- Wrap most of you main script's code within ``if __name__ == '__main__':`` block,
  to make sure it doesn't run again (most likely generating error) when each worker
  process is launched. You can place your dataset and :class:`~torch.utils.data.DataLoader`
  instance creation logic here, as it doesn't need to be re-executed in workers.

- Make sure that any custom :attr:`collate_fn`, :attr:`worker_init_fn`
  or :attr:`dataset` code is declared as top level definitions, outside of the
  ``__main__`` check. This ensures that they are available in worker processes.
  (this is needed since functions are pickled as references only, not ``bytecode``.)

.. _data-loading-randomness:

Randomness in multi-process data loading
""""""""""""""""""""""""""""""""""""""""""

By default, each worker will have its PyTorch seed set to ``base_seed + worker_id``,
where ``base_seed`` is a long generated by main process using its RNG (thereby,
consuming a RNG state mandatorily). However, seeds for other libraries may be
duplicated upon initializing workers (e.g., NumPy), causing each worker to return
identical random numbers. (See :ref:`this section <dataloader-workers-random-seed>` in FAQ.).

In :attr:`worker_init_fn`, you may access the PyTorch seed set for each worker
with either :func:`torch.utils.data.get_worker_info().seed <torch.utils.data.get_worker_info>`
or :func:`torch.initial_seed()`, and use it to seed other libraries before data
loading.

Memory Pinning
--------------

Host to GPU copies are much faster when they originate from pinned (page-locked)
memory. See :ref:`cuda-memory-pinning` for more details on when and how to use
pinned memory generally.

For data loading, passing :attr:`pin_memory=True` to a
:class:`~torch.utils.data.DataLoader` will automatically put the fetched data
Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled
GPUs.

The default memory pinning logic only recognizes Tensors and maps and iterables
containing Tensors.  By default, if the pinning logic sees a batch that is a
custom type (which will occur if you have a :attr:`collate_fn` that returns a
custom batch type), or if each element of your batch is a custom type, the
pinning logic will not recognize them, and it will return that batch (or those
elements) without pinning the memory.  To enable memory pinning for custom
batch or data type(s), define a :meth:`pin_memory` method on your custom
type(s).

See the example below.

Example::

    class SimpleCustomBatch:
        def __init__(self, data):
            transposed_data = list(zip(*data))
            self.inp = torch.stack(transposed_data[0], 0)
            self.tgt = torch.stack(transposed_data[1], 0)

        # custom memory pinning method on custom type
        def pin_memory(self):
            self.inp = self.inp.pin_memory()
            self.tgt = self.tgt.pin_memory()
            return self

    def collate_wrapper(batch):
        return SimpleCustomBatch(batch)

    inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    dataset = TensorDataset(inps, tgts)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                        pin_memory=True)

    for batch_ndx, sample in enumerate(loader):
        print(sample.inp.is_pinned())
        print(sample.tgt.is_pinned())


.. autoclass:: DataLoader
.. autoclass:: Dataset
.. autoclass:: IterableDataset
.. autoclass:: TensorDataset
.. autoclass:: ConcatDataset
.. autoclass:: ChainDataset
.. autoclass:: BufferedShuffleDataset
.. autoclass:: Subset
.. autofunction:: torch.utils.data.get_worker_info
.. autofunction:: torch.utils.data.random_split
.. autoclass:: torch.utils.data.Sampler
.. autoclass:: torch.utils.data.SequentialSampler
.. autoclass:: torch.utils.data.RandomSampler
.. autoclass:: torch.utils.data.SubsetRandomSampler
.. autoclass:: torch.utils.data.WeightedRandomSampler
.. autoclass:: torch.utils.data.BatchSampler
.. autoclass:: torch.utils.data.distributed.DistributedSampler