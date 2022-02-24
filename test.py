from free_group import free_group_bounded, normal_closure, reciprocal, conjugation

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
