"""Simple example script."""

from lib.example_file import example_function


def main():
    number = 7
    print(f"{number} times two is {example_function(number)}")


if __name__ == "__main__":
    main()
