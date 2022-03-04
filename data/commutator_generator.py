from .closure_generator import generator_from_normal_closure_v2 as generator_from_closure
from core.free_group import commutator


def generator_from_commutator(group_index = 2):
    while True:
        acc = next(generator_from_closure([1], group_index))
        for t in range(2, group_index + 1):
            acc = commutator(acc, next(generator_from_closure([t], group_index)))
        yield acc
