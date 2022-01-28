def test1(a):
    print('test1')
    return a
def test2(a):
    print('test2')
    return a

if test1(False) and test2(True):
    print('pass')




class Foo():
    def __init__(self, func):
        self.a=1
        self.func=func
    def prt(self):
        print(self.a)
    def apply(self):
        #self.a=self.func(self.a)
        self.func(self)

def plus1(self):
    self.a+=1

foo=Foo(plus1)
foo.prt()
foo.apply()
foo.prt()
