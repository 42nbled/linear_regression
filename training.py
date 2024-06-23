import pandas as pd
import matplotlib.pyplot as plt

def write_parameters(file_path, theta0, theta1):
    with open(file_path, 'w') as file:
        file.write(f"{theta0}\n")
        file.write(f"{theta1}\n")

def normalize(data):
    km_mean = data['km'].mean()
    km_std = data['km'].std()
    price_mean = data['price'].mean()
    price_std = data['price'].std()
    
    data['km_normalized'] = (data['km'] - km_mean) / km_std
    data['price_normalized'] = (data['price'] - price_mean) / price_std
    
    return data, km_mean, km_std, price_mean, price_std

def training(data, learning_rate, iteration):
    views = []

    theta0 = 0
    theta1 = 0

    x_values = data['km_normalized'].to_numpy()
    y_values = data['price_normalized'].to_numpy()

    for i in range(iteration):
        total_diff_theta0 = 0
        total_diff_theta1 = 0
        size = len(data)

        for km, price in zip(x_values, y_values):
            current_diff = (theta0 * km + theta1) - price
            total_diff_theta0 += current_diff * km
            total_diff_theta1 += current_diff
        
        theta0 -= total_diff_theta0 / size * learning_rate
        theta1 -= total_diff_theta1 / size * learning_rate

        line_y_normalized = theta0 * x_values + theta1
        
        views.append((x_values, line_y_normalized))

    parameter_file = 'parameters.txt'
    write_parameters(parameter_file, theta0, theta1)

    return views, theta0, theta1

def denormalize_line(x_values, line_y_normalized, km_mean, km_std, price_mean, price_std):
    x_values_denormalized = x_values * km_std + km_mean
    line_y_denormalized = line_y_normalized * price_std + price_mean
    return x_values_denormalized, line_y_denormalized

def display_plot(views, data, km_mean, km_std, price_mean, price_std, a, b):
    current_view = [0]

    def update_plot(view_index):
        ax.clear()
        x_values, line_y_normalized = views[view_index]
        x_values_denormalized, line_y_denormalized = denormalize_line(
            x_values, line_y_normalized, km_mean, km_std, price_mean, price_std
        )
        
        ax.plot(x_values_denormalized, line_y_denormalized, label=f'Iteration {view_index + 1}')
        
        ax.scatter(data['km'], data['price'], color='red', label='Data Points   ')
        
        ax.set_title('Car Price vs Kilometers Driven')
        ax.set_xlabel('Kilometers Driven')
        ax.set_ylabel('Price')
        
        ax.legend()
        plt.draw()

    fig, ax = plt.subplots()
    update_plot(current_view[0])

    def on_key(event):
        if event.key == ' ':
            current_view[0] = (current_view[0] + 1) % len(views)
            update_plot(current_view[0])

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

def main():
    data = pd.read_csv('data.csv')
    data, km_mean, km_std, price_mean, price_std = normalize(data)
    views, a, b = training(data, 0.01, 1000)
    display_plot(views, data, km_mean, km_std, price_mean, price_std, a, b)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
