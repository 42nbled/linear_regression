def read_parameters(file_path):
    with open(file_path, 'r') as file:
        theta0 = float(file.readline().strip())
        theta1 = float(file.readline().strip())
    return theta0, theta1

def predict_price(mileage, theta0, theta1):
    return theta0 + mileage * theta1

def main():
    parameter_file = 'parameters.txt'

    try:
        theta0, theta1 = read_parameters(parameter_file)
    except Exception:
        theta0, theta1 = 0, 0

    try:
        mileage = float(input("Enter the mileage of the car: "))
        estimated_price = predict_price(mileage, theta0, theta1)
        print(f"The estimated price of the car for {mileage} miles is: ${estimated_price:.2f}")
    except ValueError:
        print("Error: Please enter a valid number for mileage.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
