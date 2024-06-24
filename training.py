import pandas as pd
import matplotlib.pyplot as plt

# Global normalization variables
km_mean = 0
km_std = 0
price_mean = 0
price_std = 0

def write_parameters(file_path, theta0, theta1):
    theta1_denormalized = theta1 * price_std / km_std
    theta0_denormalized = price_mean + theta0 * price_std - theta1_denormalized * km_mean
    with open(file_path, 'w') as file:
        file.write(f"{theta0_denormalized}\n")
        file.write(f"{theta1_denormalized}\n")

def normalize(data):
    global km_mean, km_std, price_mean, price_std
    km_mean = data['km'].mean()
    km_std = data['km'].std()
    price_mean = data['price'].mean()
    price_std = data['price'].std()
    
    data['km_normalized'] = (data['km'] - km_mean) / km_std
    data['price_normalized'] = (data['price'] - price_mean) / price_std
    
    return data

def denormalize_line(x_values, line_y_normalized):
    global km_mean, km_std, price_mean, price_std
    x_values_denormalized = x_values * km_std + km_mean
    line_y_denormalized = line_y_normalized * price_std + price_mean
    return x_values_denormalized, line_y_denormalized

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
            current_diff = (theta0 + theta1 * km) - price
            total_diff_theta0 += current_diff
            total_diff_theta1 += current_diff * km
        
        theta0 -= total_diff_theta0 / size * learning_rate
        theta1 -= total_diff_theta1 / size * learning_rate

        line_y_normalized = theta0 + theta1 * x_values
        
        views.append((x_values, line_y_normalized))

    parameter_file = 'parameters.txt'
    write_parameters(parameter_file, theta0, theta1)

    return views

def display_plot(views, data):
    current_view = [0]

    def update_plot(view_index):
        ax.clear()
        x_values, line_y_normalized = views[view_index]
        x_values_denormalized, line_y_denormalized = denormalize_line(x_values, line_y_normalized)
        
        ax.plot(x_values_denormalized, line_y_denormalized, label=f'Iteration {view_index + 1}')
        
        ax.scatter(data['km'], data['price'], color='red', label='Data Points')
        
        ax.set_title('Car Price vs Kilometers Driven')
        ax.set_xlabel('Kilometers Driven')
        ax.set_ylabel('Price')
        
        ax.legend()
        plt.draw()

    fig, ax = plt.subplots()
    update_plot(current_view[0])

    def on_key(event):
        if event.key == ' ':
            current_view[0] = (current_view[0] + 10) % len(views)
            update_plot(current_view[0])

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

def main():
    data = pd.read_csv('data.csv')
    data = normalize(data)
    views = training(data, 0.001, 10000)
    display_plot(views, data)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
