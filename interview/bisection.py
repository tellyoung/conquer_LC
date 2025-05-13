def search(list_n, x):
    n = len(list_n)
    p, q = 0, n
    while p <= q:
        mid = (p + q) // 2
        if list_n[mid] == x:
            print("find x idx is ", mid)
            return list_n
        elif list_n[mid] > x:
            q = mid
        else:
            p = mid
        # print(mid)


if __name__ == "__main__":
    a = [i for i in range(10)]
    print(a)
    search(a, 8)