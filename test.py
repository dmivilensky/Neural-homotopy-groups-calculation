from core.free_group import reciprocal, conjugation, is_in_subgroup
from core.generators import generator_from_free_group_bounded as free_group_bounded, generator_from_normal_closure as normal_closure

a = next(free_group_bounded())
b = next(free_group_bounded())

x = [1]
y = [2]

print(a)
print(reciprocal(a))

print(a, b)
print("a^b", conjugation(a, b))
print(conjugation(x, y))

print(next(normal_closure([a])))
print(next(normal_closure([x])))

def test_is_in_subgroup():
    subgroup = [1, 2]

    assert is_in_subgroup(subgroup, [1, 2])
    assert is_in_subgroup(subgroup, [-2, 1, 2, 2])
    assert is_in_subgroup(subgroup, [2, 1])
    assert is_in_subgroup(subgroup, [2, 1, 2, 1, 2, 1])
    assert not is_in_subgroup(subgroup, [2, 1, 2, 1, 2, 2])

    subgroup = [1, 2, 3]
    assert is_in_subgroup(subgroup, [2, 3, 1])
    assert is_in_subgroup(subgroup, [-3, 2, 3, 1, 3])
