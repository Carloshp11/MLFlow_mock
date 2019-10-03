def couple_numbers(b_, c_):
    return b_ + 10 ** 5 * round(c_, 3)


def decouple_numbers(bc):
    c_ = round(bc / 10 ** 5, 3)
    b_ = round(bc - c_ * 10 ** 5, 3)
    return b_, c_


if __name__ == '__main':
    from random import randint
    n = 1000

    rnd = [randint() for x in range(n)]
    rnd2 = [randint(0, n) for x in range(n)]
    rnd10 = [(x - n/2) / 10 for x in rnd]
    rnd20 = [(x - n/2) / 10 for x in rnd2]


    def check(b, c):
        A = decouple_numbers(couple_numbers(b, c))
        B1 = round(b, 3)
        B2 = round(c, 3)
        return A, (B1, B2), (A[0] == B1, A[1] == B2)


    checklist = [check(a, b) for a, b in zip(rnd10, rnd20)]
    ll = [l[2] for l in checklist]

    all([l[0] and l[1] for l in ll])
    falses = [i for i in range(len(ll)) if ll[i][0] and ll[i][1]]
    print('Errores sobre la prueba de {} pares de números aleatorios:'.format(n))
    print([checklist[i] for i in falses])
