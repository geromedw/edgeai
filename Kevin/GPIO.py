from periphery import GPIO

red = GPIO("/dev/gpiochip0", 6, "out")  #pin 13
yellow = GPIO("/dev/gpiochip2", 9, "out") #pin 16
green = GPIO("/dev/gpiochip4", 10, "out") #pin 18

def changeLed(word):
    if word == "boom":
        print("boom")
        print("LED1")
        red.write(True)
        print()
    elif word == "hond":
        print("kat")
        red.write(False)
        print()
