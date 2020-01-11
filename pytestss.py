import inspect
import re
import unittest
import math
import pytest

class InsufficientException(Exception):
    pass

class MobileInventory():

    def __init__(self,inventory={}):
        #mi = MobileInventory()
        if not isinstance(inventory,dict):
            raise TypeError("Input inventory must be a dictionary")
        #type1 = set(type(i) for i in inventory.keys())
        #print(set(map(type,inventory)))
        #set(map(type,inventory)) == {str}
        if [i for i in inventory.keys() if not isinstance(i, str)]:
            raise ValueError("Mobile model name must be a string")
        if [i for i in inventory.values() if not isinstance(i,int) ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if [i for i in inventory.values() if i < 0 ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if inventory is not {}:
            self.balance_inventory = inventory
        else:
            self.balance_inventory = {}

    def add_stock(self,new_stock):
        if not isinstance(new_stock,dict):
            raise TypeError("Input inventory must be a dictionary")
        if [i for i in new_stock.keys() if not isinstance(i, str)]:
            raise ValueError("Mobile model name must be a string")
        if [i for i in new_stock.values() if not isinstance(i,int) ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if [i for i in new_stock.values() if i < 0 ] :
            raise ValueError("No. of mobiles must be a positive integer")
        #if new_stock.keys() in self.balance_inventory.keys():
        for k,v in new_stock.items():
            if k in self.balance_inventory.keys():
                self.balance_inventory[k] = v + self.balance_inventory[k]
            else:
                self.balance_inventory[k] = v
        #print("new stock: {0}".format(self.balance_inventory))

    def sell_stock(self,requested_stock):
        if not isinstance(requested_stock,dict):
            raise TypeError("Input inventory must be a dictionary")
        if [i for i in requested_stock.keys() if not isinstance(i, str)]:
            raise ValueError("Mobile model name must be a string")
        if [i for i in requested_stock.values() if not isinstance(i,int) ] :
            raise ValueError("No. of mobiles must be a positive integer")
        if [i for i in requested_stock.values() if i < 0 ]  :
            raise ValueError("No. of mobiles must be a positive integer")
        for k,v in requested_stock.items():
            if k in self.balance_inventory.keys():
                if self.balance_inventory[k] < v:
                    raise InsufficientException("Insufficient Stock")
                else:
                    self.balance_inventory[k] = self.balance_inventory[k] - v
            else:
                raise InsufficientException("No Stock. New Model Request")

        #print("new stock: {0}".format(self.balance_inventory))

class TestingInventoryCreation():
    def test_creating_empty_inventory(self):
        m = MobileInventory()
        assert m.balance_inventory == {}

    def test_creating_specified_inventory(self):
        m = MobileInventory({'iPhone Model X':100, 'Xiaomi Model Y': 1000, 'Nokia Model Z':25})
        assert m.balance_inventory == {'iPhone Model X':100, 'Xiaomi Model Y': 1000, 'Nokia Model Z':25}

    def test_creating_inventory_with_list(self):
        with pytest.raises(TypeError) as e:
            m = MobileInventory(['iPhone Model X', 'Xiaomi Model Y', 'Nokia Model Z'])
        assert "Input inventory must be a dictionary" in str(e)

    def test_creating_inventory_with_numeric_keys(self):
        with pytest.raises(ValueError) as e:
            m = MobileInventory({1:'iPhone Model X', 2:'Xiaomi Model Y', 3:'Nokia Model Z'})
        assert "Mobile model name must be a string" in str(e)

    def test_creating_inventory_with_nonnumeric_values(self):
        with pytest.raises(ValueError) as e:
            m = MobileInventory({'iPhone Model X':'100', 'Xiaomi Model Y': '1000', 'Nokia Model Z':'25'})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_creating_inventory_with_negative_value(self):
        with pytest.raises(ValueError) as e:
            m = MobileInventory({'iPhone Model X':-45, 'Xiaomi Model Y': 200, 'Nokia Model Z':25})
        assert "No. of mobiles must be a positive integer" in str(e)




class TestInventoryAddStock():

    
    @classmethod
    def setup_class(cls):
        cls.m =  MobileInventory( {'iPhone Model X':100, 'Xiaomi Model Y': 1000, 'Nokia Model Z':25})


    def test_add_new_stock_as_dict(self):
        self.m.add_stock({'iPhone Model X':50, 'Xiaomi Model Y': 2000, 'Nokia Model A':10})
        assert self.m.balance_inventory == {'iPhone Model X':150, 'Xiaomi Model Y': 3000, 'Nokia Model Z':25, 'Nokia Model A':10}

    def test_add_new_stock_as_list(self):
        with pytest.raises(TypeError) as e:
            self.m.add_stock(['iPhone Model X', 'Xiaomi Model Y', 'Nokia Model Z'])
        assert "Input inventory must be a dictionary" in str(e)

    def test_add_new_stock_with_numeric_keys(self):
        with pytest.raises(ValueError) as e:
            self.m.add_stock({1:'iPhone Model A', 2:'Xiaomi Model B', 3:'Nokia Model C'})
        assert "Mobile model name must be a string" in str(e)

    def test_add_new_stock_with_nonnumeric_values(self):
        with pytest.raises(ValueError) as e:
            self.m.add_stock({'iPhone Model A':'50', 'Xiaomi Model B':'2000', 'Nokia Model C':'25'})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_add_new_stock_with_float_values(self):
        with pytest.raises(ValueError) as e:
            self.m.add_stock({'iPhone Model A':50.5, 'Xiaomi Model B':2000.3, 'Nokia Model C':25})
        assert "No. of mobiles must be a positive integer" in str(e)


class TestInventorySellStock():


    @classmethod
    def setup_class(cls):
        cls.m =  MobileInventory( {'iPhone Model A':50, 'Xiaomi Model B': 2000, 'Nokia Model C':10, 'Sony Model D':1})


    def test_sell_stock_as_dict(self):
        self.m.sell_stock({'iPhone Model A':2, 'Xiaomi Model B':20, 'Sony Model D':1})
        assert self.m.balance_inventory == {'iPhone Model A':48, 'Xiaomi Model B': 1980, 'Nokia Model C':10, 'Sony Model D':0}

    def test_sell_stock_as_list(self):
        with pytest.raises(TypeError) as e:
            self.m.add_stock(['iPhone Model A', 'Xiaomi Model B', 'Nokia Model C'] )
        assert "Input inventory must be a dictionary" in str(e)

    def test_sell_stock_with_numeric_keys(self):
        with pytest.raises(ValueError) as e:
            self.m.sell_stock({1:'iPhone Model A', 2:'Xiaomi Model B', 3:'Nokia Model C'} )
        assert "Mobile model name must be a string" in str(e)

    def test_sell_stock_with_nonnumeric_values(self):
        with pytest.raises(ValueError) as e:
            self.m.sell_stock({'iPhone Model A':'2', 'Xiaomi Model B':'3', 'Nokia Model C':'4'})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_sell_stock_with_float_values(self):
        with pytest.raises(ValueError) as e:
            self.m.sell_stock({'iPhone Model A':2.5, 'Xiaomi Model B':3.1, 'Nokia Model C':4})
        assert "No. of mobiles must be a positive integer" in str(e)

    def test_sell_stock_of_nonexisting_model(self):
        with pytest.raises(InsufficientException) as e:
            self.m.sell_stock({'iPhone Model B':2, 'Xiaomi Model B':5} )
        assert "No Stock. New Model Request" in str(e)

    def test_sell_stock_of_insufficient_stock(self):
        with pytest.raises(InsufficientException) as e:
            self.m.sell_stock({'iPhone Model A':2, 'Xiaomi Model B':5, 'Nokia Model C': 15})
        assert "Insufficient Stock" in str(e)



        #m1 = MobileInventory({'iPhone Model X':100, 'Xiaomi Model Y': 0, 'Nokia Model Z':25})
#m1.add_stock({'iPhone Model X':100})
#m1.sell_stock({'iPhone Model X':100})