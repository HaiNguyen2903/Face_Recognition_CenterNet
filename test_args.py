class Parent:
  def __init__(self, txt):
    self.message = txt

  def printmessage(self):
    print(self.message)

class Child(Parent):
  def __init__(self, txt, txt2):
    super().__init__(txt)
    self.txt2 = txt2

  def print2(self):
  	print(self.text2)

x = Child("Hello, and welcome!", 'text2')
x.print2()

x.printmessage()
