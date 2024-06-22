import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def training(data):
    views = []

    a = 0
    b = 0

    x_values = data['km'].to_numpy()
    # y_values = data['price'].to_numpy()

    for i in range(10):
        total_diff = 0
        size = 0
        for row in data.itertuples(index=True, name='Pandas'):
            km = row[1]
            price = row[2]
            current_diff = (a * km + b) - price
            total_diff += current_diff
            size += 1
        print(f" ")
        total_diff /= size
        b -= total_diff
        print(f"{total_diff}")
        line_y = a * x_values + b
        views.append((x_values, line_y))

    return views

def display_plot(views, data):
    current_view = [0]

    def update_plot(view_index):
        ax.clear()
        ax.plot(views[view_index][0], views[view_index][1], label=f'View {view_index}')
        
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
            current_view[0] = (current_view[0] + 1) % len(views)
            update_plot(current_view[0])

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

def main():
    data = pd.read_csv('data.csv')
    views = training(data)
    display_plot(views, data)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
