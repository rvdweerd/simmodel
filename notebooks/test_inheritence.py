class Animal(object): #object not needed in Python3
    def __init__(self):
        print('Created animal')
        self.age=0
        self.type='Animal'
    def pet():
        print('...')
    def pet2():
        print('pet2')
    def pet3(self):
        print('pet3')
    def getAge(self):
        print(self.age)

class Dog(Animal):
    def __init__(self, name='unnamed'):
        self.name=name
        #super().__init__()
    def pet(self):
        print('woof')

class SomethingElse():
    pass


doggie = Dog('brutus')
doggie.pet()
doggie.getAge()
s = SomethingElse()
k=0
#policy_a = PolicyA
#policy_a.reset()