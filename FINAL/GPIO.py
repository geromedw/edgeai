#EdgeAI
#Team AI
#Moens Ralph, De Wilde Gerome, Lahey Kevin

#Import
from periphery import GPIO

#Variable
red = GPIO("/dev/gpiochip0", 6, "out")  #pin 13
yellow = GPIO("/dev/gpiochip2", 9, "out") #pin 16
green = GPIO("/dev/gpiochip4", 10, "out") #pin 18

#Function
def changeLed(word):
    if word == "boom":
        red.write(True)
    elif word == "deur":
        red.write(False)
    elif word == "hond":
        yellow.write(True)
    elif word == "tafel":
        yellow.write(False)
    elif word == "vrede":
        green.write(True)
    elif word == "water":
        green.write(False)

def turnOff():
    red.write(False)
    yellow.write(False)
    green.write(False)