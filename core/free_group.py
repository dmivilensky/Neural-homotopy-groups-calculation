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


def minimal_change_to_identity(word):
    inf = 10 ** 9
    n = len(word)

    dp = [[inf for d in range(0, n - l + 1)] for l in range(n)]
    for l in range(n):
        dp[l][0] = 0
        dp[l][1] = 0 if word[l] == 0 else 1
    
    for d in range(2, n + 1):
        for l in range(n - d + 1):
            choice = [
                dp[l + 1][d - 1] if word[l] == 0 else inf,
                dp[l][d - 1] if word[l + d - 1] == 0 else inf,
                dp[l + 1][d - 2] if word[l] == -word[l + d - 1] else inf,
                1 + dp[l + 1][d - 2], # change one of word[l], word[r] to make them reciprocal
                1 + dp[l][d - 1], # change word[r] to 0
                1 + dp[l + 1][d - 1], # change word[l] to 0
                *[dp[l][split - l] + dp[split][l + d - split] for split in range(l, l + d)]
            ]
            dp[l][d] = min(choice)

    return dp[0][n]


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
