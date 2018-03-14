class Foo:
    def __init__(self):
        self.x = 0

    @property
    def y(self):
        return self.x+10

foo = Foo()
print(foo.x)
print(foo.y)
foo.x = 10
print(foo.y)
foo.y = 20