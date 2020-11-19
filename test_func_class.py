def prt_hi():
    print('hi')
    
class MyClass:
    def __init__(self, foo):
        if foo != 1:
            raise ValueError("foo is not equal to 1!")
        self.foo = foo
            
    def foo_plus_1(self):
        self.foo += 1
        
    def prt_foo(self):