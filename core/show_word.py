import string

def cap_inverse_to_string(element_to_inverse):
    return f'{element_to_inverse}^{{-1}}'

def minus_inverse_to_string(element_to_inverse):
    return f'(-{element_to_inverse})'


def word_to_string(word, inverse_to_string = "cap"):
    if inverse_to_string == "cap":
        inverse_to_string = cap_inverse_to_string
    elif inverse_to_string == "minus":
        inverse_to_string = minus_inverse_to_string
    else:
        raise ValueError
    return ''.join(
        map(lambda w: string.ascii_lowercase[abs(w) - 1] if w > 0 else inverse_to_string(string.ascii_lowercase[abs(w) - 1]), word)
    )
