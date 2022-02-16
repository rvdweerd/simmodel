class A1():
    def __init__(self):
        self.a0=10
        self.a=self.a0
    def increment(self,obj):
        self.a+=1
        print('incr  A1, a=',self.a)
        obj.reset()
    def reset(self):
        self.a=self.a0
        print('reset A1, a=',self.a)
    def print(self):
        print('print A1, a=',self.a)

class B1(A1):
    def __init__(self):
        super(B1,self).__init__()
        self.b0=100
        self.b=self.b0
    def increment(self):
        self.b+=1
        super().increment(super())
        print('incr  B1, b=',self.b)
    def reset(self):
        self.b=self.b0
        print('reset B1, b=',self.b)
    def print(self):
        print('print B1, b=',self.b)



# a1=A1()
# a1.print()
# a1.increment(a1)
# a1.reset()

b1=B1()
b1.print()
b1.increment()
b1.reset()
