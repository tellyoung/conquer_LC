class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = None

def func(head):
    if head == None:
        return 
    p, q = head, head.next
    while q:
        tmp = q.next
        q.next = p
        if p == head:
            p.next = None
        p = q
        q = tmp
    return p

if __name__ == "__main__":
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n1.next = n2
    n2.next = n3

    h = func(n1)

    p = h
    while p:
        print(p.value)
        p = p.next