class UserRequest():
    def __init__(self):
        #atributs, gets and sets, methods
        self.coffe_type = input("What type of coffee would you like? (Espresso, Latte, Cappuccino): ")
        self.size = input("What size would you like? (Small, Medium, Large): ")

class Storage():
    def __init__(self):
        #atributs
        self.coffe_types = ["Espresso", "Latte", "Cappuccino"]
        self.sizes = ["Small", "Medium", "Large"]

class Cash():
    def __init__(self):
        #atributs
        self.amount = float(input("Please enter the amount you are paying: "))