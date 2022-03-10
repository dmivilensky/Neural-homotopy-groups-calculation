from random import sample, random
from math import asinh, cosh
from .free_group import reciprocal, conjugation, normalize


##
#   Structures are used to generate words from free groups
##
#   gen1 = ClosureGenerator(generators_number=2, length_config = {'max-length': 50}, subgroup=[1])
#   gen2 = ClosureGenerator(generators_number=2, length_config = {'max-length': 50})
#   gen = ChoiceGenerator((gen1, 0.6), (gen2, 0.4)) 
##
#   gen now generates with 0.6 chance element from normal closure of [1] and with chance 0.4 random element from F = <x, y>
##
#   for word in gen.take(10):
#       print(word)
##
#   This code will print ten words from gen
##


##
#   This abastraction is used to encode how to generate length for elements.
#   'length' means that all generated elements will be with the given length
#   'max-length' means that length will be generated using hyperbolic (?) distribution in range of 1 to max-length
#   'dist' means that length will be generated using given distribution
##

class LengthConfig:
    LENGTH = 'length'
    MAX_LENGTH = 'max-length'
    DIST = 'dist'

    @classmethod
    def read_length_config(cls, length_config):
        if cls.LENGTH in length_config:
            return lambda: length_config[cls.LENGTH]
        elif cls.MAX_LENGTH in length_config:
            return lambda: max(1, int(asinh(random() * cosh(length_config[cls.MAX_LENGTH] - 1))))
        elif cls.DIST in length_config:
            return length_config[cls.DIST]
        else:
            raise ValueError('incorrect `length_config`')

##
#   These methods are helpers for WordGenerators
##

def generate_from_free_group(generators_number, length_config):
    generators = set(range(-generators_number, generators_number + 1)) - set([0])
    length_generator = LengthConfig.read_length_config(length_config)
    word, length = [], length_generator()

    for _ in range(length):
        factor = sample(generators - set(reciprocal(word[-1:])), 1)[0]
        word.append(factor)
        
    return word


def generate_from_normal_closure(subgroup, generators_number, length_config):
    length_generator = LengthConfig.read_length_config(length_config)
    word, length = [], length_generator()

    while len(word) < length:
        factor = subgroup if random() > 0.5 else reciprocal(subgroup)
            
        conjugator = generate_from_free_group(
            generators_number=generators_number,
            length_config={LengthConfig.MAX_LENGTH : (length - len(word) - len(factor)) // 2}
        )
        word += conjugation(factor, conjugator)

    return normalize(word)


##
#   These classes are used to build generators of words from free group
##

class WordGenerator:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __call__(self):
        return self.__next__()

    def take(self, count):
        return BoundedGenerator(self, count)

    def append(self, generator):
        return SequenceGenerator(self, generator)         

class ChoiceGenerator(WordGenerator):
    def __init__(self, *args):
        self.generators = list(args)

    def __next__(self):
        rand_point, current = random(), 0
        for gen, proba in self.generators:
            if current + proba >= rand_point:
                return gen()
        return self.generators[-1][0]()

class ClosureGenerator(WordGenerator):
    def __init__(self, generators_number, length_config, subgroup = None):
        self.generators_number = generators_number
        self.length_config = length_config
        self.subgroup = subgroup

    def __next__(self):
        return generate_from_free_group(self.generators_number, self.length_config) if self.subgroup is None else generate_from_normal_closure(self.subgroup, self.generators_number, self.length_config)

class BoundedGenerator(WordGenerator):
    def __init__(self, generator, count):
        self.generator, self.to_generate = generator, count

    def __next__(self):
        if self.to_generate >= 0:
            self.to_generate -= 1
            return self.generator()
        else:
            raise StopIteration

class SequenceGenerator(WordGenerator):
    def __init__(self, *args):
        self.sequence = list(args)

    def __next__(self):
        while self.sequence:
            try:
                return self.sequence[0]()
            except StopIteration:
                self.sequence.pop(0)
        raise StopIteration

    def append(self, generator):
        self.sequence.append(generator)
