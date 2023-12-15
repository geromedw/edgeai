import argparse 
from periphery import GPIO

led1 = GPIO("/dev/gpiochip2", 13, "out")

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--boom',
        action='store_true',
        help="boom")
    parser.add_argument(
        '--kat',
        action='store_true',
        help="kat")
    parser.add_argument(
        '--hond',
        action='store_true',
        help="hond")
    parser.add_argument(
        '--water',
        action='store_true',
        help="water")
    parser.add_argument(
        '--vrede',
        action='store_true',
        help="vrede")    
    parser.add_argument(
        '--uitlaat',
        action='store_true',
        help="uitlaat")
    
    return parser.parse_args()

def main():
    args = parse_command_line_args()

    if args.boom:
        print("boom")
        print("LED1")
        led1.write(True)
        print()
    elif args.kat:
        print("kat")
        led1.write(False)
        print()
    elif args.hond:
        print("hond")
        print("LED3")
        print()
    elif args.water:
        print("water")
        print("LED4")
        print()
    elif args.vrede:
        print("vrede")
        print("LED5")
        print()
    elif args.uitlaat:
        print("uitlaat")
        print("LED6")
        print()

if __name__ == "__main__":
    main()