from lsfunc import *

A = [['ID', 'Source', 'Drink'], 
     [0, 'Hops', 'Beer'],
     [1, 'Apples', 'Cider']]
      
B = [['ID', 'Source', 'Drink'], 
     [10, 'Grapes', 'Wine']]


a = listtable()
b = listtable()
a.setList(A)
b.setList(B)
c = a.union(b)

