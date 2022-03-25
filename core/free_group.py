def reciprocal(word):
    return [-factor for factor in word[::-1]]


def normalize(word):
    normalized = []

    for factor in word:
        if factor == 0:
            continue
        if len(normalized) == 0:
            normalized.append(factor)
            continue

        if factor == -normalized[-1]:
            normalized.pop()
        else:
            normalized.append(factor)

    return normalized


def is_cyclic_permutation(a, b):
    if len(a) != len(b):
        return False

    double_b = b * 2
    for i in range(2 * len(b)):
        if double_b[i] == a[0] and double_b[i:i + len(a)] == a:
            return True
    return False


def is_from_normal_closure(generator, word):
    contained_smth_to_reduce = True
    generator_len = len(generator)

    while contained_smth_to_reduce:
        contained_smth_to_reduce = False
        new_word = []

        i = 0
        while i <= len(word) - generator_len:
            subword = word[i:i + generator_len]
            if is_cyclic_permutation(subword, generator) or is_cyclic_permutation(subword, reciprocal(generator)):
                contained_smth_to_reduce = True
                i += generator_len
            else:
                new_word.append(word[i])
                i += 1
        
        if i < len(word):
            new_word += word[-(len(word)-i):]
        word = normalize(new_word)
    
    return len(word) == 0


def conjugation(word, conjugator):
    return reciprocal(conjugator) + word + conjugator


def commutator(x, y):
    return reciprocal(x) + reciprocal(y) + x + y


def symmetric_commutator(words):
    acc = words[0]
    for w in words[1:]:
        acc = commutator(acc, w)
    return acc


def word_as_str(word):
    letters = "xyzpqrstuvwklmn"
    return "".join(map(lambda factor: letters[abs(factor) - 1] + ("⁻¹" if factor < 0 else ""), word))


def print_word(word):
    print(word_as_str(word))
