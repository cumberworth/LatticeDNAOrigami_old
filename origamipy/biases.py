"""Bias functions."""


STACK_TAG = 'numstackedpairs'


class StackingBias:
    def __init__(self, stack_energy, stack_mult):
        self._stack_energy = stack_energy
        self._stack_mult = stack_mult
        self._complementary_stack_mult = 1 - stack_mult

    def __call__(self, order_params):
        total_stack_energy = order_params[STACK_TAG]*self._stack_energy
        return -total_stack_energy*self._complementary_stack_mult

    @property
    def fileformat_value(self):
        return self._stack_mult


class TotalBias:
    def __init__(self, biases):
        self._biases = biases

    def __call__(self, order_params):
        return sum([bias(order_params) for bias in self._biases])
