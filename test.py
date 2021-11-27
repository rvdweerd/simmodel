class Animal(object):
    def __init__(self, yell):
        self.yell=yell
    def pet(self):
        print(self.yell)

class Dog(Animal):
    def __init__(self):
        super().__init__("woof")
    def pet(self):
        super().pet()
        print('!')


#dog = Animal("woof")
dog = Dog()
dog.pet()