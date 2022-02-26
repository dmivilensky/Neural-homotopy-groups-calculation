from free_group import free_group_bounded, normal_closure, reciprocal, conjugation, is_in_subgroup

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

    subgroup = [1, 2, 3]
    assert is_in_subgroup(subgroup, [2, 3, 1])
    assert is_in_subgroup(subgroup, [-3, 2, 3, 1, 3])
