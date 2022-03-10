from core.generators import ClosureGenerator, SequenceGenerator
from core.free_group import reciprocal, conjugation, commutator, is_in_subgroup

gen1 = ClosureGenerator(generators_number=2, length_config={'length': 5})
gen2 = ClosureGenerator(generators_number=3, length_config={'length': 3}, subgroup=[1])

for word in SequenceGenerator(gen1.take(10), gen2.take(5)):
    print(word)

a, b = gen1(), gen2()
print(a, b)

print(reciprocal(a), reciprocal(b))

print(conjugation(a, b))

print(commutator(a, b))

print(is_in_subgroup([1], a), is_in_subgroup([1], b))
