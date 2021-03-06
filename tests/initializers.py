"""Matrix initialization utilities."""
import abc
import numpy as np
import six

class Initializer(six.with_metaclass(abc.ABCMeta)):
  """Defines API for a weight initializer."""

  def __init__(self):
    """Initialization API for weight initializer.

    This method can be overridden to save input
    keyword arguments for the specific initializer.
    """
    pass

  @abc.abstractmethod
  def __call__(self, shape):
    pass


class Uniform(Initializer):

  def __init__(self, low=0.0, high=1.0):
    super(Uniform, self).__init__()
    self.low = low
    self.high = high

  def __call__(self, shape):
    return np.reshape(np.random.uniform(
        self.low, self.high, np.prod(shape)), shape)


class Range(Initializer):

  def __call__(self, shape):
    # NOTE: We offset the initial values by 1 s.t. none of the
    # weights are zero valued to begin with.
    return np.reshape(np.arange(np.prod(shape)) + 1, shape)
